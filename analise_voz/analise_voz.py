import sys
import numpy as np
import librosa


ARQUIVO = "Arlindo.wav"
LIMIAR_SILENCIO_DB = 30


def carregar_voz(caminho, sr=22050, offset=0.0, duracao=None):
    """Carrega o audio em mono e remove os trechos de silencio."""
    y, sr = librosa.load(caminho, sr=sr, mono=True, offset=offset, duration=duracao)
    blocos = librosa.effects.split(y, top_db=LIMIAR_SILENCIO_DB)
    if len(blocos):
        voz = np.concatenate([y[ini:fim] for ini, fim in blocos])
    else:
        voz = y
    return y, voz, sr


def nota(freq):
    return librosa.hz_to_note(freq)


def analisar_pitch(voz, sr):
    f0, _, _ = librosa.pyin(voz, fmin=60, fmax=500, sr=sr, frame_length=2048)
    f0 = f0[~np.isnan(f0)]

    mediana = np.median(f0)
    media = np.mean(f0)
    p1, p5 = np.percentile(f0, 1), np.percentile(f0, 5)
    p95, p99 = np.percentile(f0, 95), np.percentile(f0, 99)
    iqr = np.percentile(f0, 75) - np.percentile(f0, 25)

    print("--- Pitch (F0) ---")
    print("F0 mediana (Hz):", round(mediana, 1), "->", nota(mediana))
    print("F0 media (Hz):", round(media, 1))
    print("Grave fala (5pct):", round(p5, 1), "->", nota(p5))
    print("Agudo fala (95pct):", round(p95, 1), "->", nota(p95))
    print("Extremo grave (1pct):", round(p1, 1), "->", nota(p1))
    print("Extremo agudo (99pct):", round(p99, 1), "->", nota(p99))
    print("Desvio padrao (Hz):", round(np.std(f0), 1))
    print("IQR (Hz):", round(iqr, 1))


def analisar_timbre(voz, sr):
    centroide = librosa.feature.spectral_centroid(y=voz, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=voz, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=voz)[0]
    zcr = librosa.feature.zero_crossing_rate(voz)[0]

    harm, perc = librosa.effects.hpss(voz)
    razao_hp = np.sum(harm ** 2) / (np.sum(perc ** 2) + 1e-9)

    print("--- Timbre ---")
    print("Centroide espectral (Hz):", round(np.mean(centroide), 1))
    print("Rolloff (Hz):", round(np.mean(rolloff), 1))
    print("Flatness:", round(np.mean(flatness), 4))
    print("ZCR:", round(np.mean(zcr), 4))
    print("Razao harmonico/percussivo:", round(razao_hp, 2))


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else ARQUIVO

    # Pitch e duracao no audio inteiro
    y, voz, sr = carregar_voz(caminho, sr=22050)
    print("Duracao total (s):", round(len(y) / sr, 1))
    print("Duracao com voz (s):", round(len(voz) / sr, 1))
    analisar_pitch(voz, sr)

    # Timbre numa janela menor para nao estourar a memoria
    _, voz_janela, sr_janela = carregar_voz(caminho, sr=16000, offset=30, duracao=120)
    analisar_timbre(voz_janela, sr_janela)


if __name__ == "__main__":
    main()
