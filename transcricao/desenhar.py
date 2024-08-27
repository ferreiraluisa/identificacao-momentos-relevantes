import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot_logmel_spectrogram_no_labels(file_path, output_path):
    # Carregar o arquivo de áudio
    y, sr = librosa.load(file_path, sr=None)

    # Calcular o espectrograma Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # Convertê-lo para escala logarítmica (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plotar o espectrograma Log-Mel sem legendas
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, fmax=8000)
    plt.axis('off')  # Remover os eixos
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
# Caminho do arquivo .wav de entrada e arquivo .png de saída
input_wav_path = 'ihsn5kZQEYA.wav'
output_png_path = 'logmel_spectrogram.png'

# Gerar e salvar o espectrograma Log-Mel
plot_logmel_spectrogram_no_labels(input_wav_path, output_png_path)
