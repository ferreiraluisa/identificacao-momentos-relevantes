import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp

# Carregar o áudio
audio_path = 'output/ihsn5kZQEYA/vocals.wav'
y, sr = librosa.load(audio_path)

# Gerar o wavegram
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Wavegram')
plt.xlabel('Tempo')
plt.ylabel('Amplitude')
plt.savefig('wavegram.png')
plt.close()


from moviepy.editor import VideoClip, ImageClip, AudioFileClip
import matplotlib.pyplot as plt

# Função para criar os frames do vídeo
def make_frame(t):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.axvline(x=t, color='r')  # Adiciona a linha vertical em 't'
    plt.title('Wavegram')
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
    
    # Salvar o frame como imagem
    plt.savefig('frame.png')
    plt.close()
    
    # Carregar a imagem e retornar como numpy array
    frame = plt.imread('frame.png')
    return frame

# Duração do áudio
duration = librosa.get_duration(y=y, sr=sr)

# Criar o vídeo a partir dos frames
video = VideoClip(make_frame, duration=duration)

# Adicionar o áudio ao vídeo
audio = AudioFileClip(audio_path)
video = video.set_audio(audio)

# Salvar o vídeo
video.write_videofile('wavegram_video.mp4', fps=24)
