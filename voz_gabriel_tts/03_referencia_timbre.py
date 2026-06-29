"""
Monta uma referência de timbre rica para o XTTS, concatenando os clipes mais
longos do Gabriel até ~target_seconds. Mais áudio de referência = clone melhor.
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path

BASE = Path(__file__).parent
WAVS = BASE / "data" / "wavs"
OUT  = BASE / "data" / "gabriel_ref.wav"

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--sr", type=int, default=24000)
args = ap.parse_args()

idx = pd.read_csv(BASE / "data" / "clips_index.csv")
idx = idx.sort_values("duration", ascending=False)  # mais longos primeiro

chunks, total = [], 0.0
for _, r in idx.iterrows():
    audio, sr = sf.read(str(WAVS / r["file_name"]))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != args.sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
    chunks.append(audio)
    total += len(audio) / args.sr
    if total >= args.seconds:
        break

ref = np.concatenate(chunks)[: int(args.seconds * args.sr)]
sf.write(str(OUT), ref, args.sr)
print(f"Referência: {len(ref)/args.sr:.1f}s @ {args.sr}Hz -> {OUT}")
