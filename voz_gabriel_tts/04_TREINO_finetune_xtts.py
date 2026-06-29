"""
Fine-tune do XTTS-v2 na voz do Gabriel (áudio JÁ limpo).
"""
import os
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse, csv, random
from pathlib import Path

BASE   = Path(__file__).parent
DATA   = BASE / "data"
META   = DATA / "metadata_clean.csv"
OUTDIR = BASE / "xtts_ft"
TRAIN  = DATA / "metadata_train.csv"
EVAL   = DATA / "metadata_eval.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--eval-frac", type=float, default=0.1)
args = ap.parse_args()

rows = [r for r in csv.reader(open(META, encoding="utf-8"), delimiter="|") if len(r) >= 2 and r[1].strip()]
random.seed(42); random.shuffle(rows)
n_eval = max(2, int(len(rows) * args.eval_frac))
eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="|")
        w.writerow(["audio_file", "text", "speaker_name"])
        for r in data:
            w.writerow([r[0], r[1], r[2] if len(r) > 2 else "gabriel"])

write_csv(TRAIN, train_rows)
write_csv(EVAL, eval_rows)
print(f"Treino: {len(train_rows)} | Eval: {len(eval_rows)}")

CACHE = Path.home() / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"
CKPT_DIR = OUTDIR / "run" / "training" / "XTTS_v2.0_original_model_files"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
for fn in ["model.pth", "vocab.json", "config.json"]:
    dst = CKPT_DIR / fn
    if not dst.exists() and (CACHE / fn).exists():
        dst.symlink_to(CACHE / fn)
        print(f"  symlink {fn} <- cache")

from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

print(f"\nIniciando fine-tune: {args.epochs} épocas, batch {args.batch}, grad-accum {args.grad_accum}")
config_path, ckpt, tokenizer, out_path, speaker_ref = train_gpt(
    language="pt",
    num_epochs=args.epochs,
    batch_size=args.batch,
    grad_acumm=args.grad_accum,
    train_csv=str(TRAIN),
    eval_csv=str(EVAL),
    output_path=str(OUTDIR),
    max_audio_length=int(11 * 22050), 
)
print("\n=== FINE-TUNE CONCLUÍDO ===")
print("config:", config_path)
print("checkpoint dir:", out_path)
print("speaker_ref sugerido:", speaker_ref)
