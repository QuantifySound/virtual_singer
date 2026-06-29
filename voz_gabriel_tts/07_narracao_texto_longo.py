"""
Narra um texto LONGO (história) com a voz fine-tuned do Gabriel.
Quebra o texto em frases, gera trecho a trecho e concatena tudo 
num único .wav com pausas naturais.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = Path(__file__).parent
WAVS = BASE / "data" / "wavs_clean"

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--file", help="arquivo .txt com a história")
g.add_argument("--text", help="texto direto")
ap.add_argument("--out", default=str(BASE / "out_finetuned" / "historia.wav"))
ap.add_argument("--ckpt-dir", default=None)
ap.add_argument("--lang", default="pt")
ap.add_argument("--nrefs", type=int, default=10)
ap.add_argument("--pause", type=float, default=0.3, help="pausa entre frases (s)")
ap.add_argument("--maxchars", type=int, default=220, help="máx. caracteres por trecho")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--repetition-penalty", type=float, default=10.0)
args = ap.parse_args()

# --- texto ---
texto = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
texto = re.sub(r"\s+", " ", texto).strip()

def split_sentences(t, maxchars):
    partes = re.split(r"(?<=[.!?…])\s+", t)
    chunks, cur = [], ""
    def flush():
        nonlocal cur
        if cur.strip():
            chunks.append(cur.strip())
        cur = ""
    for p in partes:
        p = p.strip()
        if not p:
            continue
        if len(p) > maxchars:
            flush()
            while len(p) > maxchars:
                corte = p.rfind(",", 0, maxchars)
                corte = corte if corte > 40 else maxchars
                chunks.append(p[:corte].strip())
                p = p[corte:].strip()
            cur = p
        elif len(cur) + len(p) + 1 <= maxchars:
            cur = (cur + " " + p).strip()
        else:
            flush()
            cur = p
    flush()
    return [c for c in chunks if c]

chunks = split_sentences(texto, args.maxchars)
print(f"Texto: {len(texto)} caracteres -> {len(chunks)} trechos")

if args.ckpt_dir:
    run = Path(args.ckpt_dir)
else:
    runs = sorted(glob.glob(str(BASE / "xtts_ft/run/training/GPT_XTTS_FT-*")))
    assert runs, "Nenhum run de treino encontrado"
    run = Path(runs[-1])
ckpt = run / "best_model.pth"
ORIG = next(c for c in [run.parent / "XTTS_v2.0_original_model_files",
                        run / "XTTS_v2.0_original_model_files"] if (c / "config.json").exists())

idx = pd.read_csv(BASE / "data" / "clips_index.csv").sort_values("duration", ascending=False)
refs = [str(WAVS / fn) for fn in idx["file_name"].head(args.nrefs)]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Modelo: {ckpt}\nDevice: {device}")
config = XttsConfig(); config.load_json(str(ORIG / "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(ORIG / "vocab.json"),
                      use_deepspeed=False)
model.to(device)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=refs, gpt_cond_len=30, max_ref_length=60)
print("Modelo carregado. Narrando...\n", flush=True)

SR = 24000
silence = np.zeros(int(args.pause * SR), dtype=np.float32)
pieces = []
for i, ch in enumerate(chunks, 1):
    out = model.inference(ch, args.lang, gpt_cond_latent, speaker_embedding,
                          temperature=args.temperature,
                          repetition_penalty=args.repetition_penalty,
                          length_penalty=1.0, enable_text_splitting=True)
    wav = np.asarray(out["wav"], dtype=np.float32)
    pieces.append(wav); pieces.append(silence)
    print(f"  [{i}/{len(chunks)}] {len(wav)/SR:4.1f}s  {ch[:60]}{'...' if len(ch)>60 else ''}", flush=True)

audio = np.concatenate(pieces)
peak = np.abs(audio).max()
if peak > 0: audio = audio * (0.97 / peak)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
sf.write(args.out, audio, SR)
print(f"\nPronto! História completa: {args.out}  ({len(audio)/SR:.1f}s)")