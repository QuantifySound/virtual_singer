"""
Pipeline de limpeza do corpus de voz:
  1. Silero VAD — filtra segmentos com pouca fala (< 30% do tempo)
  2. ResembleEnhance denoise — remove ruído residual do Demucs
"""

import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

BASE       = Path(__file__).parent
INPUT_DIR  = BASE / "work_voice" / "corpus_segments"
OUTPUT_DIR = BASE / "work_voice" / "corpus_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SPEECH_RATIO = 0.30   # descarta segmentos com < 30% de fala

print(f"Dispositivo: {DEVICE}")
print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}\n")

print("Carregando Silero VAD...")
try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    vad_model = load_silero_vad()
    vad_model = vad_model.to(DEVICE)
    USE_NEW_API = True
except ImportError:
    # fallback: torch.hub
    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    get_speech_timestamps_hub, _, read_audio, _, _ = vad_utils
    USE_NEW_API = False


def get_speech_ratio(wav_path: Path) -> float:
    """Retorna fração do áudio que contém fala (0.0–1.0)."""
    try:
        if USE_NEW_API:
            wav = read_audio(str(wav_path))  # resample para 16kHz internamente
            wav = wav.to(DEVICE)
            ts = get_speech_timestamps(wav, vad_model, return_seconds=True)
            speech_dur = sum(t["end"] - t["start"] for t in ts)
            total_dur  = len(wav) / 16000
        else:
            wav = read_audio(str(wav_path), sampling_rate=16000)
            ts  = get_speech_timestamps_hub(wav, vad_model, sampling_rate=16000)
            speech_dur = sum(t["end"] - t["start"] for t in ts) / 16000
            total_dur  = len(wav) / 16000
        return speech_dur / total_dur if total_dur > 0 else 0.0
    except Exception:
        return 0.0

print("Carregando ResembleEnhance...")
try:
    from resemble_enhance.enhancer.inference import denoise as re_denoise
    HAS_RESEMBLE = True
    print("  ResembleEnhance OK.")
except ImportError:
    HAS_RESEMBLE = False
    print("  ResembleEnhance não encontrado — só aplicará VAD (sem denoise).")


def enhance_audio(data: np.ndarray, sr: int) -> np.ndarray:
    """Aplica ResembleEnhance denoise. Retorna numpy array."""
    if not HAS_RESEMBLE:
        return data
    wav_t = torch.from_numpy(data).float().to(DEVICE)
    with torch.no_grad():
        denoised, new_sr = re_denoise(wav_t, sr, DEVICE)
    return denoised.squeeze().cpu().numpy()

segments = sorted(INPUT_DIR.glob("seg_*.wav"))
print(f"\n{len(segments)} segmentos para processar.\n")

kept    = 0
dropped = 0

for seg in tqdm(segments, desc="Limpando corpus"):
    ratio = get_speech_ratio(seg)

    if ratio < MIN_SPEECH_RATIO:
        dropped += 1
        continue

    # carrega, denoise, salva
    data, sr = sf.read(str(seg), dtype="float32")
    clean    = enhance_audio(data, sr)
    sf.write(str(OUTPUT_DIR / seg.name), clean, sr)
    kept += 1
    
total     = kept + dropped
kept_min  = kept * 8 / 60
total_min = total * 8 / 60

print(f"\n{'='*50}")
print(f"Segmentos totais:    {total}")
print(f"Mantidos (≥30% fala): {kept}  ({kept_min:.1f} min)")
print(f"Descartados (<30%):  {dropped}  ({dropped * 8 / 60:.1f} min)")
print(f"Taxa de aproveitamento: {kept/total*100:.1f}%")
print(f"\nCorpus limpo: {OUTPUT_DIR}")
print(f"Próximo passo: fine-tune do Seed-VC com os {kept} segmentos em corpus_clean/")
