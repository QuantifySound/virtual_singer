"""
Mistura o vocal convertido do Arlindo com o instrumental do Cafajeste.
Saída: music_work/final_mix.wav
"""

import argparse
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

BASE       = Path(__file__).parent
MUSIC_WORK = BASE / "music_work"

INST_WAV  = MUSIC_WORK / "instrumental.wav"
OUT_WAV   = MUSIC_WORK / "final_mix.wav"

parser = argparse.ArgumentParser()
parser.add_argument("--vocal-path", default=None,
                    help="Caminho do vocal convertido (padrão: music_work/arlindo_vocal.wav)")
parser.add_argument("--vocal-vol", type=float, default=1.0,
                    help="Ganho do vocal (1.0 = sem alteração)")
parser.add_argument("--inst-vol",  type=float, default=1.0,
                    help="Ganho do instrumental (1.0 = sem alteração)")
args = parser.parse_args()

VOCAL_WAV = Path(args.vocal_path) if args.vocal_path else MUSIC_WORK / "arlindo_vocal.wav"

vocal, sr_v = sf.read(str(VOCAL_WAV))
inst,  sr_i = sf.read(str(INST_WAV))

if sr_v != sr_i:
    print(f"[AVISO] Sample rates diferentes: vocal={sr_v}Hz, inst={sr_i}Hz")
    print("  Resampleando instrumental para o SR do vocal...")
    import scipy.signal
    inst = scipy.signal.resample(inst, int(len(inst) * sr_v / sr_i))

# Garante mono/stereo compatível
if vocal.ndim == 1 and inst.ndim == 2:
    vocal = np.stack([vocal, vocal], axis=1)
elif vocal.ndim == 2 and inst.ndim == 1:
    inst = np.stack([inst, inst], axis=1)

# Alinha tamanhos (corta o mais longo)
n = min(len(vocal), len(inst))
vocal = vocal[:n]
inst  = inst[:n]

mix = vocal * args.vocal_vol + inst * args.inst_vol

# Normaliza pra evitar clipping
peak = np.abs(mix).max()
if peak > 0.98:
    mix = mix * (0.98 / peak)
    print(f"  Normalizado (pico era {peak:.2f})")

sf.write(str(OUT_WAV), mix, sr_v)
info = sf.info(str(OUT_WAV))
print(f"Mix final: {OUT_WAV}  ({info.duration:.1f}s @ {sr_v}Hz)")
print(f"  vocal×{args.vocal_vol}  instrumental×{args.inst_vol}")
