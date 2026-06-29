"""
Inferência com o XTTS-v2 FINE-TUNED na voz do Gabriel.
Gera as mesmas frases do zero-shot para comparação.
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse, glob
from pathlib import Path
import pandas as pd
import torch, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

BASE = Path(__file__).parent
WAVS = BASE / "data" / "wavs_clean"
OUTDIR = BASE / "out_finetuned"; OUTDIR.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", default=None, help="pasta do run de treino (default: mais recente)")
ap.add_argument("--text", default=None)
ap.add_argument("--lang", default="pt")
ap.add_argument("--nrefs", type=int, default=10)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--repetition-penalty", type=float, default=10.0)
args = ap.parse_args()

# localiza o run de treino e o best_model.pth
if args.ckpt_dir:
    run = Path(args.ckpt_dir)
else:
    runs = sorted(glob.glob(str(BASE / "xtts_ft/run/training/GPT_XTTS_FT-*")))
    assert runs, "Nenhum run de treino encontrado em xtts_ft/run/training/"
    run = Path(runs[-1])
ckpt = run / "best_model.pth"
assert ckpt.exists(), f"best_model.pth não encontrado em {run}"

def _find_orig():
    for cand in [run.parent / "XTTS_v2.0_original_model_files",
                 run / "XTTS_v2.0_original_model_files"]:
        if (cand / "config.json").exists():
            return cand
    raise FileNotFoundError("XTTS_v2.0_original_model_files (config.json) não encontrado")
ORIG = _find_orig()
config_path = ORIG / "config.json"
vocab_path  = ORIG / "vocab.json"
print(f"Arquivos base: {ORIG}")

frases = [args.text] if args.text else [
    "Olá, eu sou o Gabriel e esta é a minha voz clonada por inteligência artificial.",
    "Ligar as luzes da sala e ajustar a temperatura para vinte e dois graus.",
    "Bom dia! Hoje o céu está limpo e a previsão é de sol o dia inteiro.",
]

idx = pd.read_csv(BASE / "data" / "clips_index.csv").sort_values("duration", ascending=False)
refs = [str(WAVS / fn) for fn in idx["file_name"].head(args.nrefs)]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Modelo fine-tuned: {ckpt}")
print(f"Device: {device} | {len(refs)} clipes de referência")

config = XttsConfig(); config.load_json(str(config_path))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(vocab_path),
                      use_deepspeed=False)
model.to(device)
print("Modelo carregado. Calculando latentes...", flush=True)

gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=refs, gpt_cond_len=30, max_ref_length=60)

for i, txt in enumerate(frases, 1):
    out = model.inference(txt, args.lang, gpt_cond_latent, speaker_embedding,
                          temperature=args.temperature,
                          repetition_penalty=args.repetition_penalty,
                          length_penalty=1.0, enable_text_splitting=True)
    path = OUTDIR / f"gabriel_ft_{i:02d}.wav"
    sf.write(str(path), out["wav"], 24000)
    print(f"[{i}] {path.name}  ({len(out['wav'])/24000:.1f}s)  <-  {txt!r}", flush=True)

print("\nPronto. Compare:")
print("  zero-shot:  out_zeroshot/gabriel_zs_XX.wav")
print("  fine-tune:  out_finetuned/gabriel_ft_XX.wav")
