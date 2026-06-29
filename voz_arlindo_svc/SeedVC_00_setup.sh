#!/bin/bash
# Setup Seed-VC para singing voice conversion zero-shot
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
SEEDVC_DIR="$BASE/seed_vc"

if [ ! -d "$SEEDVC_DIR" ]; then
    echo "=== Clonando Seed-VC ==="
    git clone https://github.com/Plachtaa/seed-vc "$SEEDVC_DIR"
else
    echo "=== Seed-VC já clonado ==="
fi

echo ""
echo "=== Instalando dependências ==="
pip install \
    einops \
    librosa \
    cached_path \
    hydra-core \
    omegaconf \
    vector_quantize_pytorch \
    vocos \
    --quiet

echo ""
echo "=== Baixando modelos ==="
python -c "
import sys
sys.path.insert(0, '$SEEDVC_DIR')
from huggingface_hub import hf_hub_download
import os

os.makedirs('$SEEDVC_DIR/checkpoints/seed-uvit-whisper-small-wavenet', exist_ok=True)
os.makedirs('$SEEDVC_DIR/checkpoints/seed-uvit-whisper-base-f0-44k', exist_ok=True)

# Modelo de voz (speech)
files_vc = [
    'config.json', 'model.safetensors',
]
for f in files_vc:
    dest = '$SEEDVC_DIR/checkpoints/seed-uvit-whisper-small-wavenet/' + f
    if not os.path.exists(dest):
        print(f'  baixando {f}...')
        hf_hub_download(
            repo_id='Plachtaa/seed-vc',
            filename='checkpoints/seed-uvit-whisper-small-wavenet/' + f,
            local_dir='$SEEDVC_DIR'
        )

# Modelo de canto (singing) — 44kHz com F0
files_sing = ['config.json', 'model.safetensors']
for f in files_sing:
    dest = '$SEEDVC_DIR/checkpoints/seed-uvit-whisper-base-f0-44k/' + f
    if not os.path.exists(dest):
        print(f'  baixando {f} (singing)...')
        hf_hub_download(
            repo_id='Plachtaa/seed-vc',
            filename='checkpoints/seed-uvit-whisper-base-f0-44k/' + f,
            local_dir='$SEEDVC_DIR'
        )
print('Modelos prontos.')
"

echo ""
echo "=== Setup Seed-VC concluído ==="
echo "Próximo passo: python SeedVC_02_inferencia.py"
