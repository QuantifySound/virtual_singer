"""
Limpa os clipes do Gabriel para o fine-tune:
  1. Filtro passa-alta 70Hz  -> remove o rumble/hum <100Hz
  2. resemble-enhance denoise -> remove ruído ambiente (preserva fonemas)
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse, glob, csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfilt
from resemble_enhance.denoiser.inference import denoise

BASE   = Path(__file__).parent
SRC    = BASE / "data" / "wavs"
DST    = BASE / "data" / "wavs_clean"
DST.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--hp", type=float, default=70.0, help="freq de corte do passa-alta (Hz)")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted(glob.glob(str(SRC / "*.wav")))
if args.limit:
    files = files[: args.limit]
print(f"Limpando {len(files)} clipes (passa-alta {args.hp}Hz + denoise) em {device}...", flush=True)

def highpass(x, sr, fc, order=4):
    sos = butter(order, fc / (sr / 2), btype="high", output="sos")
    return sosfilt(sos, x).astype(np.float32)

for i, f in enumerate(files, 1):
    a, sr = sf.read(f)
    if a.ndim == 2:
        a = a.mean(axis=1)
    a = highpass(a.astype(np.float32), sr, args.hp)
    dwav = torch.from_numpy(a).float()
    out, sr_out = denoise(dwav, sr, None, device)   
    out = out.cpu().numpy()
    # normaliza pico para -1dBFS
    peak = np.abs(out).max()
    if peak > 0:
        out = out * (0.95 / peak)
    sf.write(str(DST / Path(f).name), out, int(sr_out))
    if i % 25 == 0 or i == len(files):
        print(f"  {i}/{len(files)}", flush=True)

# reescreve metadata apontando para wavs_clean
src_meta = BASE / "data" / "metadata.csv"
dst_meta = BASE / "data" / "metadata_clean.csv"
if src_meta.exists():
    with open(src_meta, encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="|"))
    with open(dst_meta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="|")
        for r in rows:
            if len(r) >= 2:
                name = Path(r[0]).name
                w.writerow([f"wavs_clean/{name}"] + r[1:])
    print(f"metadata limpo: {dst_meta}")

print(f"Pronto. Clipes limpos em {DST}")
