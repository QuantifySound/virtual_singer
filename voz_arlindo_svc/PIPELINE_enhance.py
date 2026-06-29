import argparse
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--output", default=None)
parser.add_argument("--lambd",  type=float, default=0.5,
                    help="0=só denoise, 1=enhance máximo (default 0.5)")
parser.add_argument("--nfe",    type=int, default=32,
                    help="Passos de difusão (default 32)")
parser.add_argument("--denoise-only", action="store_true")
args = parser.parse_args()

input_path  = Path(args.input).resolve()
output_path = Path(args.output).resolve() if args.output else \
              input_path.parent / (input_path.stem + "_enhanced.wav")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Resemble Enhance ===  {input_path.name} → {output_path.name}")
print(f"  device={device}  lambd={args.lambd}  nfe={args.nfe}")

audio, sr = sf.read(str(input_path))
if audio.ndim == 2:
    audio = audio.mean(axis=1)
dwav = torch.from_numpy(audio).float()

if args.denoise_only:
    from resemble_enhance.denoiser.inference import denoise
    out, sr_out = denoise(dwav, sr, device)
else:
    from resemble_enhance.enhancer.inference import enhance
    out, sr_out = enhance(dwav, sr, device, nfe=args.nfe, lambd=args.lambd, tau=0.5)

out_np = out.cpu().numpy() if hasattr(out, "cpu") else np.array(out)
sf.write(str(output_path), out_np, int(sr_out))
info = sf.info(str(output_path))
print(f"Salvo: {output_path}  ({info.duration:.1f}s @ {int(sr_out)}Hz)")
