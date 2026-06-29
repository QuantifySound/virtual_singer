"""
Transcreve os clipes do Gabriel com Whisper large-v3-turbo (PT-BR).
Gera data/metadata.csv no formato esperado pelo fine-tune do XTTS
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import glob
import csv
import torch
from pathlib import Path
from transformers import pipeline

BASE = Path(__file__).parent
WAVS = BASE / "data" / "wavs"
OUT  = BASE / "data" / "metadata.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--speaker", default="gabriel")
args = ap.parse_args()

files = sorted(glob.glob(str(WAVS / "*.wav")))
if args.limit:
    files = files[: args.limit]

print(f"Transcrevendo {len(files)} clipes com whisper-large-v3-turbo...", flush=True)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               device=0, torch_dtype=torch.float16, batch_size=16)

rows = []
for i, f in enumerate(files, 1):
    r = asr(f, generate_kwargs={"language": "portuguese", "task": "transcribe"})
    text = r["text"].strip()
    rows.append((f"wavs/{Path(f).name}", text, args.speaker))
    if i % 25 == 0 or i == len(files):
        print(f"  {i}/{len(files)}", flush=True)

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="|")
    for row in rows:
        w.writerow(row)

print(f"\nTranscrição salva: {OUT}  ({len(rows)} linhas)")
print("Amostras:")
for row in rows[:5]:
    print("  ", row[0], "->", repr(row[1]))
