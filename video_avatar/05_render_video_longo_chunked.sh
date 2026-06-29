#!/bin/bash
set -u
cd /workspace/desafio_final/Sonic
PY="./venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/workspace/desafio_final
IMG="$BASE/Sonic/arlindo_pics/Arlindo_Deixe_Me_Ir.png"
VOC="$BASE/music_batch/02/vocal_seedvc.wav"
MIX="$BASE/music_batch/02/mix_seedvc.wav"
TMP="$BASE/avatar_out/tmp"; mkdir -p "$TMP" "$BASE/avatar_out/safety" "$BASE/avatar_out/final"

mux () { ffmpeg -y -i "$1" -i "$2" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "$3" 2>/dev/null; }

FINAL="$BASE/avatar_out/final/deixe_me_ir.mp4"
if [ -f "$FINAL" ]; then
  echo "[skip] $FINAL já existe"
else
  DUR=$(python3 -c "import soundfile as sf; print(sf.info('$VOC').duration)")
  SEG=57
  echo "=== deixe_me_ir 16:9 ORIGINAL: ${DUR}s em segmentos de ${SEG}s ==="
  LIST="$TMP/deixe_concat.txt"; : > "$LIST"
  i=0; start=0
  while python3 -c "import sys; sys.exit(0 if $start < $DUR else 1)"; do
    segvoc="$TMP/deixe_seg_${i}_voc.wav"; segout="$TMP/deixe_seg_${i}.mp4"
    ffmpeg -y -ss "$start" -t "$SEG" -i "$VOC" "$segvoc" 2>/dev/null
    if [ ! -f "$segout" ]; then
      echo "  -- segmento $i (start ${start}s) --"
      $PY render_one.py --image "$IMG" --vocal "$segvoc" --out "$segout" || { echo "FALHA no segmento $i"; exit 1; }
    fi
    echo "file '$segout'" >> "$LIST"
    i=$((i+1)); start=$((start+SEG))
  done
  ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$TMP/deixe_169_concat.mp4" 2>/dev/null
  mux "$TMP/deixe_169_concat.mp4" "$MIX" "$FINAL"
  echo "[ok] CLIPE COMPLETO 16:9 (original) -> $FINAL"
fi

OUT="$BASE/avatar_out/safety/deixe_me_ir_crop_wide.mp4"
if [ -f "$OUT" ]; then
  echo "[skip] $OUT já existe"
else
  echo "=== deixe_me_ir CROP large/wide (expand=1.2) ==="
  $PY render_one.py --image "$IMG" --vocal "$VOC" --out "$TMP/deixe_wide_raw.mp4" --crop --expand 1.2 \
    && mux "$TMP/deixe_wide_raw.mp4" "$MIX" "$OUT" && echo "[ok] $OUT"
fi

echo "=== RECUPERAÇÃO (original + large) CONCLUÍDA ==="
