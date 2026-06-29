"""Separa vocal e instrumental de uma música usando Demucs htdemucs_ft."""
from pathlib import Path
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

BASE = Path(__file__).parent
WORK = BASE / "music_work_telo"
INPUT_WAV = WORK / "telo_full.wav"
VOCAL_OUT = WORK / "vocals_original.wav"
INST_OUT  = WORK / "instrumental.wav"

print(f"GPU: {torch.cuda.get_device_name(0)}")
model = get_model("htdemucs_ft").cuda().eval()
print(f"Stems: {model.sources}")
vocals_idx = model.sources.index("vocals")

data, sr = sf.read(str(INPUT_WAV), dtype="float32", always_2d=True)
wav_t = torch.from_numpy(data.T).unsqueeze(0).cuda()

with torch.no_grad():
    sources = apply_model(model, wav_t, device="cuda", shifts=1, split=True, overlap=0.25, progress=True)

vocals = sources[0, vocals_idx].cpu().numpy().T
# Instrumental = soma de todos os stems que não são vocal
inst_stems = [sources[0, i].cpu().numpy().T for i in range(sources.shape[1]) if i != vocals_idx]
instrumental = sum(inst_stems)

sf.write(str(VOCAL_OUT), vocals, sr)
sf.write(str(INST_OUT), instrumental, sr)
print(f"\n✓ Vocal       : {VOCAL_OUT}  ({len(vocals)/sr:.1f}s @ {sr}Hz)")
print(f"✓ Instrumental: {INST_OUT}  ({len(instrumental)/sr:.1f}s @ {sr}Hz)")
