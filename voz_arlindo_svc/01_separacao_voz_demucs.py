"""
Roda Demucs htdemucs_ft via API Python
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

BASE       = Path(__file__).parent
WAVS_DIR   = BASE / "work_voice" / "wavs"
DEMUCS_OUT = BASE / "demucs_output" / "htdemucs_ft"
CORPUS_DIR = BASE / "work_voice" / "corpus_segments"
WORK_DIR   = BASE / "work_voice"

if not torch.cuda.is_available():
    print("[ERRO] CUDA não disponível.")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

from demucs.pretrained import get_model
from demucs.apply import apply_model

print("Carregando htdemucs_ft...")
model = get_model("htdemucs_ft")
model = model.cuda()
model.eval()
vocals_idx = model.sources.index("vocals")
print(f"Stems disponíveis: {model.sources}  |  vocals_idx={vocals_idx}\n")

wavs = sorted(WAVS_DIR.glob("*.wav"))
print(f"{len(wavs)} arquivos WAV encontrados.")

DEMUCS_OUT.mkdir(parents=True, exist_ok=True)

for wav_path in wavs:
    out_dir     = DEMUCS_OUT / wav_path.stem
    out_vocals  = out_dir / "vocals.wav"

    if out_vocals.exists():
        print(f"  [já existe] {wav_path.name}")
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Processando: {wav_path.name}")

    # Carrega áudio
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # data: [samples, channels] → tensor: [1, channels, samples]
    wav_t = torch.from_numpy(data.T).unsqueeze(0)
    if wav_t.shape[1] == 1:          # mono → estéreo (demucs espera 2 canais)
        wav_t = wav_t.repeat(1, 2, 1)
    wav_t = wav_t.cuda()

    with torch.no_grad():
        sources = apply_model(
            model, wav_t,
            device="cuda",
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True,
        )
    # sources: [batch=1, stems, channels=2, samples]

    vocals = sources[0, vocals_idx]
    vocals_mono = vocals.mean(0).cpu().numpy()

    sf.write(str(out_vocals), vocals_mono, sr)
    print(f"  -> {out_vocals}  ({len(vocals_mono)/sr:.1f}s)")

print("\n=== Demucs concluído ===")

import subprocess
print("\n=== Concatenando stems de voz ===")

all_vocals = sorted((DEMUCS_OUT).glob("*/vocals.wav"))
arlindo    = [v for v in all_vocals if v.parent.name == "Arlindo"]
others     = [v for v in all_vocals if v.parent.name != "Arlindo"]
ordered    = arlindo + others

concat_list = WORK_DIR / "concat_list.txt"
with open(concat_list, "w") as f:
    for v in ordered:
        f.write(f"file '{v}'\n")

corpus_raw = WORK_DIR / "corpus_raw.wav"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-ar", "44100", "-ac", "1",
    str(corpus_raw)
], check=True, capture_output=True)

r = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "csv=p=0", str(corpus_raw)],
    capture_output=True, text=True
)
dur = float(r.stdout.strip())
print(f"  Corpus bruto: {dur:.0f}s  ({dur/60:.1f} min)")

print("\n=== Segmentando em chunks de 8s para o treino ===")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

subprocess.run([
    "ffmpeg", "-y", "-i", str(corpus_raw),
    "-f", "segment", "-segment_time", "8",
    "-ar", "44100", "-ac", "1",
    str(CORPUS_DIR / "seg_%04d.wav")
], check=True, capture_output=True)

n = len(list(CORPUS_DIR.glob("seg_*.wav")))
print(f"  {n} segmentos gerados.")
print(f"\nCorpus: {CORPUS_DIR}")
print("Próximo passo: fine-tune do Seed-VC.")
