"""
Renderiza os avatares finais do Arlindo cantando (Sonic), em lote.
Carrega o modelo UMA vez e processa todos os jobs.

Para cada job:
  1. Recorta o trecho do VOCAL convertido (driver de lip-sync) e do MIX.
  2. Detecta+recorta o rosto da imagem temática (crop).
  3. Sonic: imagem + vocal -> vídeo (lip-sync).
  4. Faz o mux do MIX completo (vocal+instrumental) no vídeo final.
"""
import os, subprocess, sys
from sonic import Sonic

BASE   = "/workspace/desafio_final"
PICS   = f"{BASE}/Sonic/arlindo_pics"
MUSIC  = f"{BASE}/music_batch"
OUT    = f"{BASE}/avatar_out/final"
TMP    = f"{BASE}/avatar_out/tmp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

JOBS = [
    ("tempo_perdido_legiao",   "01", 124, 15,   "Arlindo_Tempo_Perdido.png"),
    ("evidencias_chitaozinho", "08",  23, 39,   "Arlindo_Sinonimos.png"),
    ("robocop_gay",            "05", 107, 67,   "Arlindo_Robocop_Gay.png"),
    ("amor_e_fe",              "04", 174, 57,   "Arlindo_Amor_F#U00e9.png"),
    ("yellow_coldplay",        "03",  35, 80,   "Arlindo_Yellow.png"),
    ("i_want_to_break_free",   "07",  40, 31,   "Arlindo_I_Want_To_Break_Free.png"),
    ("deixe_me_ir",            "02",   0, None, "Arlindo_Deixe_Me_Ir.png"),
]

def ff(args):
    subprocess.run(["ffmpeg", "-y", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cut(src, dst, start, dur):
    a = ["-ss", str(start)]
    if dur is not None:
        a += ["-t", str(dur)]
    ff([*a, "-i", src, dst])

print("Carregando Sonic (uma vez)...", flush=True)
pipe = Sonic(0)
print("Sonic carregado.\n", flush=True)

results = []
for i, (name, song, start, dur, img) in enumerate(JOBS, 1):
    print(f"\n{'='*60}\n[{i}/{len(JOBS)}] {name}  (song {song}, {start}s +{dur or 'inteira'})\n{'='*60}", flush=True)
    final_mp4 = f"{OUT}/{name}.mp4"
    if os.path.exists(final_mp4):
        print(f"  já existe, pulando: {final_mp4}", flush=True)
        results.append((name, "SKIP", final_mp4)); continue
    try:
        voc_full = f"{MUSIC}/{song}/vocal_seedvc.wav"
        mix_full = f"{MUSIC}/{song}/mix_seedvc.wav"
        voc_seg  = f"{TMP}/{name}_vocal.wav"
        mix_seg  = f"{TMP}/{name}_mix.wav"
        cut(voc_full, voc_seg, start, dur)
        cut(mix_full, mix_seg, start, dur)

        img_path = f"{PICS}/{img}"
        face_info = pipe.preprocess(img_path, expand_ratio=0.5)
        print("  face_info:", face_info, flush=True)
        if face_info["face_num"] <= 0:
            raise RuntimeError("nenhum rosto detectado na imagem")

        raw_mp4 = f"{TMP}/{name}_raw.mp4"
        pipe.process(img_path, voc_seg, raw_mp4,
                     min_resolution=512, inference_steps=25, dynamic_scale=1.0)

        # mux do mix completo
        final_mp4 = f"{OUT}/{name}.mp4"
        ff(["-i", raw_mp4, "-i", mix_seg, "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest", final_mp4])
        results.append((name, "OK", final_mp4))
        print(f"  [{i}] CONCLUÍDO -> {final_mp4}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        results.append((name, f"ERRO: {e}", ""))
        print(f"  [{i}] FALHOU: {e}", flush=True)

print(f"\n{'='*60}\nRESUMO\n{'='*60}")
for name, status, path in results:
    print(f"  {status:10} {name}  {path}")
