"""
Clonagem zero-shot do Gabriel com XTTS-v2 — versão robusta.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import glob
from pathlib import Path
import pandas as pd
import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE   = Path(__file__).parent
WAVS   = BASE / "data" / "wavs"
MODEL  = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
OUTDIR = BASE / "out_zeroshot"
OUTDIR.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--text", default=None)
ap.add_argument("--lang", default="pt")
ap.add_argument("--nrefs", type=int, default=10, help="nº de clipes de referência")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--repetition-penalty", type=float, default=10.0)
args = ap.parse_args()

frases = [args.text] if args.text else [
    "Olá, eu sou o Gabriel e esta é a minha voz clonada por inteligência artificial.",
    "Ligar as luzes da sala e ajustar a temperatura para vinte e dois graus.",
    "Bom dia! Hoje o céu está limpo e a previsão é de sol o dia inteiro.",
]

idx = pd.read_csv(BASE / "data" / "clips_index.csv").sort_values("duration", ascending=False)
refs = [str(WAVS / fn) for fn in idx["file_name"].head(args.nrefs)]
print(f"Referência: {len(refs)} clipes ({idx['duration'].head(args.nrefs).sum():.1f}s no total)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Carregando XTTS-v2 de {MODEL.name}...", flush=True)

config = XttsConfig()
config.load_json(str(MODEL / "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=str(MODEL), eval=True)
model.to(device)
print("Modelo carregado. Calculando latentes de condicionamento...", flush=True)

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=refs, gpt_cond_len=30, max_ref_length=60
)

for i, txt in enumerate(frases, 1):
    out = model.inference(
        txt, args.lang, gpt_cond_latent, speaker_embedding,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        length_penalty=1.0,
        enable_text_splitting=True,   
    )
    path = OUTDIR / f"gabriel_zs_{i:02d}.wav"
    sf.write(str(path), out["wav"], 24000)
    dur = len(out["wav"]) / 24000
    print(f"[{i}] {path.name}  ({dur:.1f}s)  <-  {txt!r}", flush=True)

print("Pronto. Áudios em:", OUTDIR)
