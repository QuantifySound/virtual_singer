"""
Pipeline em batch: converte várias músicas do YouTube para a voz do Arlindo
com Seed-VC fine-tuned.
"""

import argparse, subprocess, sys, re
from pathlib import Path
import numpy as np
import soundfile as sf

BASE     = Path(__file__).parent
OUTROOT  = BASE / "music_batch"
SEEDVC_INFER = BASE / "SeedVC_02_inferencia.py"
SEEDVC_CKPT = BASE / "seed_vc/runs/arlindo_ft/ft_model.pth"
SEEDVC_CFG  = BASE / "seed_vc/runs/arlindo_ft/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--urls", help="arquivo .txt com uma URL por linha")
g.add_argument("--url", nargs="+", help="uma ou mais URLs direto")
ap.add_argument("--seedvc-pitch", type=float, default=0)
ap.add_argument("--no-denoise", action="store_true", help="não aplica redução de ruído no vocal")
ap.add_argument("--hp", type=float, default=70.0, help="freq de corte do passa-alta (Hz)")
ap.add_argument("--skip-existing", action="store_true", help="pula etapas com saída já existente")
args = ap.parse_args()

# --- lista de URLs ---
if args.urls:
    urls = [l.strip() for l in Path(args.urls).read_text().splitlines() if l.strip() and not l.startswith("#")]
else:
    urls = args.url
print(f"{len(urls)} música(s) para processar | modelo: Seed-VC FT\n")

def slug(s):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")[:40] or "musica"

def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)

_demucs = {}
def separate(src_wav, voc_out, inst_out):
    if not _demucs:
        import torch
        from demucs.pretrained import get_model
        _demucs["torch"] = torch
        _demucs["model"] = get_model("htdemucs_ft").cuda().eval()
        _demucs["vidx"] = _demucs["model"].sources.index("vocals")
    import torch
    from demucs.apply import apply_model
    model, vidx = _demucs["model"], _demucs["vidx"]
    data, sr = sf.read(str(src_wav), dtype="float32", always_2d=True)
    wav_t = torch.from_numpy(data.T).unsqueeze(0).cuda()
    with torch.no_grad():
        sources = apply_model(model, wav_t, device="cuda", shifts=1, split=True, overlap=0.25, progress=True)
    vocals = sources[0, vidx].cpu().numpy().T
    inst = sum(sources[0, i].cpu().numpy().T for i in range(sources.shape[1]) if i != vidx)
    sf.write(str(voc_out), vocals, sr)
    sf.write(str(inst_out), inst, sr)
    return sr

def denoise_vocal(in_wav, out_wav, hp):
    """Passa-alta + resemble-enhance denoise no vocal separado."""
    import torch
    from scipy.signal import butter, sosfilt
    from resemble_enhance.denoiser.inference import denoise
    a, sr = sf.read(str(in_wav))
    if a.ndim == 2:
        a = a.mean(axis=1)
    a = a.astype(np.float32)
    if hp and hp > 0:
        sos = butter(4, hp / (sr / 2), btype="high", output="sos")
        a = sosfilt(sos, a).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out, sr_out = denoise(torch.from_numpy(a).float(), sr, None, dev)
    out = out.cpu().numpy()
    peak = np.abs(out).max()
    if peak > 0:
        out = out * (0.97 / peak)
    sf.write(str(out_wav), out, int(sr_out))

def mix(vocal_wav, inst_wav, out_wav):
    voc, srv = sf.read(str(vocal_wav))
    ins, sri = sf.read(str(inst_wav))
    if srv != sri:
        import scipy.signal
        ins = scipy.signal.resample(ins, int(len(ins) * srv / sri)); sri = srv
    if voc.ndim == 1 and ins.ndim == 2: voc = np.stack([voc, voc], 1)
    if voc.ndim == 2 and ins.ndim == 1: ins = np.stack([ins, ins], 1)
    n = min(len(voc), len(ins))
    m = voc[:n] + ins[:n]
    peak = np.abs(m).max()
    if peak > 0.98: m = m * (0.98 / peak)
    sf.write(str(out_wav), m, srv)

results = []
for i, url in enumerate(urls, 1):
    d = OUTROOT / f"{i:02d}_pending"
    print(f"\n{'='*60}\n[{i}/{len(urls)}] {url}\n{'='*60}")
    try:
        OUTROOT.mkdir(exist_ok=True)
        d = OUTROOT / f"{i:02d}"
        d.mkdir(exist_ok=True)
        src_wav = d / "source.wav"
        # 1. download (pula se já existe)
        if not (args.skip_existing and src_wav.exists()):
            tmp = OUTROOT / f"{i:02d}_dl"
            run(["yt-dlp", "-x", "--audio-format", "wav", "-o", str(tmp) + ".%(ext)s", url])
            next(OUTROOT.glob(f"{i:02d}_dl.*")).replace(src_wav)
        else:
            print(" > download: já existe, pulando")
        # 2. separar (pula se já existe)
        voc, inst = d / "vocals.wav", d / "instrumental.wav"
        if not (args.skip_existing and voc.exists() and inst.exists()):
            print(" > Demucs..."); separate(src_wav, voc, inst)
        else:
            print(" > Demucs: já existe, pulando")
        # 3. redução de ruído no vocal (entrada da conversão)
        if args.no_denoise:
            voc_in = voc
        else:
            voc_in = d / "vocals_clean.wav"
            if not (args.skip_existing and voc_in.exists()):
                print(" > Denoise (passa-alta + resemble)..."); denoise_vocal(voc, voc_in, args.hp)
            else:
                print(" > Denoise: já existe, pulando")
        # 4. Seed-VC FT
        out_sv = d / "vocal_seedvc.wav"
        if not (args.skip_existing and out_sv.exists() and (d / "mix_seedvc.wav").exists()):
            run([sys.executable, str(SEEDVC_INFER), "--source", voc_in, "--output", out_sv,
                 "--checkpoint", SEEDVC_CKPT, "--config", SEEDVC_CFG, "--pitch", args.seedvc_pitch])
            mix(out_sv, inst, d / "mix_seedvc.wav")
        else:
            print(" > Seed-VC: já existe, pulando")
        results.append((i, url, "OK", str(d)))
        print(f" > [{i}] CONCLUÍDO -> {d}")
    except Exception as e:
        results.append((i, url, f"ERRO: {e}", str(d)))
        print(f" > [{i}] FALHOU: {e}")

print(f"\n{'='*60}\nRESUMO\n{'='*60}")
for i, url, status, d in results:
    print(f"  [{i}] {status:8} {d}")
