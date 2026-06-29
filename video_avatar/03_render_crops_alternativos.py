"""
Renders de SEGURANÇA em CROPs diferentes (512², só rosto) para as músicas em
que o 16:9 sem-crop pode ficar ruim. Roda DEPOIS do batch oficial.
"""
import os, subprocess
from sonic import Sonic

BASE  = "/workspace/desafio_final"
PICS  = f"{BASE}/Sonic/arlindo_pics"
MUSIC = f"{BASE}/music_batch"
OUT   = f"{BASE}/avatar_out/safety"
TMP   = f"{BASE}/avatar_out/tmp"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)

JOBS = [
    ("deixe_me_ir",          "02",   0, None, "Arlindo_Deixe_Me_Ir.png"),
    ("robocop_gay",          "05", 107, 67,   "Arlindo_Robocop_Gay.png"),
    ("amor_e_fe",            "04", 174, 57,   "Arlindo_Amor_F#U00e9.png"),
    ("yellow_coldplay",      "03",  35, 80,   "Arlindo_Yellow.png"),
    ("i_want_to_break_free", "07",  40, 31,   "Arlindo_I_Want_To_Break_Free.png"),
]
VARIANTS = [("tight", 0.3), ("medium", 0.7), ("wide", 1.2)]

def ff(args):
    subprocess.run(["ffmpeg", "-y", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cut(src, dst, start, dur):
    a = ["-ss", str(start)] + (["-t", str(dur)] if dur is not None else [])
    ff([*a, "-i", src, dst])

print("Carregando Sonic (uma vez)...", flush=True)
pipe = Sonic(0)
print("Sonic carregado.\n", flush=True)

total = len(JOBS) * len(VARIANTS); done = 0
for name, song, start, dur, img in JOBS:
    img_path = f"{PICS}/{img}"
    voc_seg = f"{TMP}/{name}_safety_vocal.wav"
    mix_seg = f"{TMP}/{name}_safety_mix.wav"
    seg_ready = False
    for vname, er in VARIANTS:
        done += 1
        final = f"{OUT}/{name}_crop_{vname}.mp4"
        print(f"\n{'='*60}\n[{done}/{total}] {name} CROP {vname} (expand_ratio={er})\n{'='*60}", flush=True)
        if os.path.exists(final):
            print(f"  já existe, pulando: {final}", flush=True); continue
        try:
            if not seg_ready:
                cut(f"{MUSIC}/{song}/vocal_seedvc.wav", voc_seg, start, dur)
                cut(f"{MUSIC}/{song}/mix_seedvc.wav",   mix_seg, start, dur)
                seg_ready = True
            face_info = pipe.preprocess(img_path, expand_ratio=er)
            print("  face_info:", face_info, flush=True)
            if face_info["face_num"] <= 0:
                raise RuntimeError("nenhum rosto detectado")
            crop_img = f"{TMP}/{name}_face_{vname}.png"
            pipe.crop_image(img_path, crop_img, face_info["crop_bbox"])
            raw = f"{TMP}/{name}_crop_{vname}_raw.mp4"
            pipe.process(crop_img, voc_seg, raw,
                         min_resolution=512, inference_steps=25, dynamic_scale=1.0)
            ff(["-i", raw, "-i", mix_seg, "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", final])
            print(f"  [{done}/{total}] CONCLUÍDO -> {final}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [{done}/{total}] FALHOU: {e}", flush=True)

print("\n=== SEGURANÇA (TODOS OS CROPS) CONCLUÍDA ===")
print("Variações em:", OUT)
