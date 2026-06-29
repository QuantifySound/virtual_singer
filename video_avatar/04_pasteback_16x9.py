"""
Paste-back: compõe o rosto animado (512x512 do Sonic) de volta na imagem
temática original (16:9), na posição do crop. Gera os vídeos finais 16:9.

Pega a bbox de cada música do log do render (render_avatars.log) e a imagem
do mapeamento. Idempotente: pula os que já existem (rode quantas vezes quiser).
"""
import os, re, subprocess

BASE = "/workspace/desafio_final"
PICS = f"{BASE}/Sonic/arlindo_pics"
FINAL = f"{BASE}/avatar_out/final"
OUT  = f"{BASE}/avatar_out/final_169"
LOG  = "/tmp/render_avatars.log"
os.makedirs(OUT, exist_ok=True)

IMG = {
    "tempo_perdido_legiao":   "Arlindo_Tempo_Perdido.png",
    "evidencias_chitaozinho": "Arlindo_Sinonimos.png",
    "robocop_gay":            "Arlindo_Robocop_Gay.png",
    "amor_e_fe":              "Arlindo_Amor_F#U00e9.png",
    "yellow_coldplay":        "Arlindo_Yellow.png",
    "i_want_to_break_free":   "Arlindo_I_Want_To_Break_Free.png",
    "deixe_me_ir":            "Arlindo_Deixe_Me_Ir.png",
}

bboxes = {}
cur = None
for line in open(LOG, encoding="utf-8", errors="ignore"):
    m = re.search(r"\[\d/7\]\s+(\w+)", line)
    if m:
        cur = m.group(1)
    b = re.search(r"crop_bbox':\s*\[([\d,\s]+)\]", line)
    if b and cur:
        bboxes[cur] = [int(x) for x in b.group(1).split(",")]

for name, img in IMG.items():
    src = f"{FINAL}/{name}.mp4"
    dst = f"{OUT}/{name}_169.mp4"
    if not os.path.exists(src):
        print(f"  pendente (sem vídeo ainda): {name}"); continue
    if os.path.exists(dst):
        print(f"  já feito: {name}"); continue
    if name not in bboxes:
        print(f"  [AVISO] sem bbox no log: {name}"); continue
    x1, y1, x2, y2 = bboxes[name]
    w, h = x2 - x1, y2 - y1
    bg = f"{PICS}/{img}"
    fc = f"[1:v]scale={w}:{h}[f];[0:v][f]overlay={x1}:{y1},scale=trunc(iw/2)*2:trunc(ih/2)*2[v]"
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", bg, "-i", src,
           "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", dst]
    print(f"  gerando 16:9: {name} (bbox {w}x{h} @ {x1},{y1})...", flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode == 0:
        print(f"    OK -> {dst}")
    else:
        print(f"    ERRO: {r.stderr.decode()[-300:]}")

print("\nVídeos 16:9 em:", OUT)
