"""
Converte vocal usando Seed-VC (zero-shot singing voice conversion).
Usa todo o corpus limpo do Arlindo como referência —> mais áudio = melhor timbre.
"""

import argparse
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

BASE       = Path(__file__).parent
SEEDVC_DIR = BASE / "seed_vc"
MUSIC_WORK = BASE / "music_work"
CORPUS_DIR = BASE / "work_voice" / "corpus_clean"

REF_WAV    = MUSIC_WORK / "arlindo_ref_seedvc.wav"
OUTPUT_DIR = MUSIC_WORK / "seedvc_out" 
OUTPUT_WAV = MUSIC_WORK / "arlindo_vocal_seedvc.wav"  

sys.path.insert(0, str(SEEDVC_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None,
                    help="Áudio fonte a converter (padrão: music_work/vocals_original.wav). "
                         "Tipicamente o vocal cru separado pelo Demucs.")
parser.add_argument("--mode",         default="singing", choices=["singing", "speech"])
parser.add_argument("--pitch",        type=float, default=0,
                    help="Pitch shift em semitons. Use 0 para manter o pitch da fonte (recipe final).")
parser.add_argument("--ref-duration", type=float, default=60,
                    help="Segundos de referência a usar (mais = melhor timbre, default 60)")
parser.add_argument("--diffusion-steps", type=int, default=30,
                    help="Passos de difusão — mais = mais qualidade e lento (default 30)")
parser.add_argument("--cfg-rate", type=float, default=0.7,
                    help="Classifier-free guidance: alto=mais 'estilo' do target, baixo=preserva fonética da fonte (default 0.7)")
parser.add_argument("--checkpoint", default=None,
                    help="Caminho de um checkpoint customizado (ex: runs/arlindo_ft/ft_model.pth)")
parser.add_argument("--config", default=None,
                    help="Config YAML do checkpoint customizado (obrigatório se --checkpoint)")
parser.add_argument("--output", default=None,
                    help="Caminho de saída do vocal convertido (default: music_work/arlindo_vocal_seedvc.wav)")
args = parser.parse_args()

INPUT_WAV = Path(args.source).resolve() if args.source else MUSIC_WORK / "vocals_original.wav"
if args.output:
    OUTPUT_WAV = Path(args.output).resolve()


def build_reference(max_seconds: float) -> np.ndarray:
    """Concatena segmentos do corpus até max_seconds para referência rica."""
    wavs = sorted(CORPUS_DIR.glob("*.wav"))
    if not wavs:
        print(f"[ERRO] Corpus vazio: {CORPUS_DIR}")
        sys.exit(1)

    chunks, total = [], 0.0
    for w in wavs:
        audio, sr = sf.read(str(w))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        chunks.append((audio, sr))
        total += len(audio) / sr
        if total >= max_seconds:
            break

    # Resampleia tudo para o SR do primeiro arquivo e concatena
    target_sr = chunks[0][1]
    import librosa
    merged = np.concatenate([
        librosa.resample(a, orig_sr=s, target_sr=target_sr) if s != target_sr else a
        for a, s in chunks
    ])
    sf.write(str(REF_WAV), merged[:int(max_seconds * target_sr)], target_sr)
    print(f"Referência: {min(total, max_seconds):.0f}s de áudio do Arlindo → {REF_WAV.name}")
    return merged, target_sr


print("=== Seed-VC: Singing Voice Conversion ===")
print(f"  Fonte: {INPUT_WAV.name}  |  Modo: {args.mode}  |  Pitch: {args.pitch:+}st  |  Referência: {args.ref_duration:.0f}s")

build_reference(args.ref_duration)

cmd = [
    sys.executable,
    str(SEEDVC_DIR / "inference.py"),
    "--source",             str(INPUT_WAV),
    "--target",             str(REF_WAV),
    "--output",             str(OUTPUT_DIR),
    "--diffusion-steps",    str(args.diffusion_steps),
    "--length-adjust",      "1.0",
    "--inference-cfg-rate", str(args.cfg_rate),
    "--fp16",               "False",
]

if args.mode == "singing":
    cmd += [
        "--f0-condition",    "True",
        "--semi-tone-shift", str(int(args.pitch)),
    ]
else:
    cmd += ["--f0-condition", "False"]

if args.checkpoint:
    cmd += ["--checkpoint", str(Path(args.checkpoint).resolve())]
if args.config:
    cmd += ["--config", str(Path(args.config).resolve())]

import subprocess
import shutil
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nRodando inferência Seed-VC...")
subprocess.run(cmd, check=True, cwd=str(SEEDVC_DIR))

generated = sorted(OUTPUT_DIR.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
if generated:
    shutil.copy2(str(generated[-1]), str(OUTPUT_WAV))
    info = sf.info(str(OUTPUT_WAV))
    print(f"\nVocal convertido: {OUTPUT_WAV}  ({info.duration:.1f}s @ {info.samplerate}Hz)")
    print("Próximo passo: python mix_final.py --vocal-path music_work/arlindo_vocal_seedvc.wav")
else:
    print("[AVISO] Arquivo de saída não encontrado — verifique erros acima")
