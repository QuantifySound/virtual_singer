"""
Worker: UM render do Sonic por processo.
Saída = vídeo do Sonic com o áudio driver (vocal) embutido.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("COQUI_TOS_AGREED", "1")
import argparse
from sonic import Sonic

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--vocal", required=True)
ap.add_argument("--out",   required=True)
ap.add_argument("--crop",  action="store_true")
ap.add_argument("--expand", type=float, default=0.5)
args = ap.parse_args()

pipe = Sonic(0)
fi = pipe.preprocess(args.image, expand_ratio=args.expand)
print("face_info:", fi, flush=True)
if fi["face_num"] <= 0:
    raise SystemExit("nenhum rosto detectado")

src = args.image
if args.crop:
    crop_path = args.out + ".crop.png"
    pipe.crop_image(args.image, crop_path, fi["crop_bbox"])
    src = crop_path

os.makedirs(os.path.dirname(args.out), exist_ok=True)
pipe.process(src, args.vocal, args.out,
             min_resolution=512, inference_steps=25, dynamic_scale=1.0)
print("OK ->", args.out, flush=True)
