# import librosa
# from librosa import display
# import numpy as np
# import IPython.display as ipd
# import matplotlib.pyplot as plt
# import soundfile as sf

# input_wav = '4Kmf4J7mROI_baixo.wav'
# output_wav = '4Kmf4J7mROI_voz.wav'
# # Carregar o arquivo de áudio
# y, sr = librosa.load(input_wav, duration=None)

# # Calcular o espectrograma magnitude e fase
# S_full, phase = librosa.magphase(librosa.stft(y))

# S_filter = librosa.decompose.nn_filter(S_full,
#                                        aggregate=np.median,
#                                        metric='cosine',
#                                        width=int(librosa.time_to_frames(2, sr=sr)))
# S_filter = np.minimum(S_full, S_filter)

# margin_i, margin_v = 2, 10
# power = 2

# mask_i = librosa.util.softmask(S_filter,
#                                margin_i * (S_full - S_filter),
#                                power=power)

# mask_v = librosa.util.softmask(S_full - S_filter,
#                                margin_v * S_filter,
#                                power=power)
# # Separar componentes
# S_foreground = mask_v * S_full
# S_background = mask_i * S_full


# # plt.figure(figsize=(12, 8))
# # plt.subplot(3, 1, 1)
# # librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
# #                          y_axis='log', sr=sr)
# # plt.title('Full spectrum')
# # plt.colorbar()

# # plt.subplot(3, 1, 2)
# # librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
# #                          y_axis='log', sr=sr)
# # plt.title('Background')
# # plt.colorbar()
# # plt.subplot(3, 1, 3)
# # librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
# #                          y_axis='log', x_axis='time', sr=sr)
# # plt.title('Foreground')
# # plt.colorbar()
# # plt.tight_layout()
# # plt.show()


# # Reconstruir a onda a partir do espectrograma
# y_foreground = librosa.istft(S_foreground * phase)

# # Salvar o resultado em um arquivo WAV
# sf.write(output_wav, y_foreground, sr)
# print(f"Vocal track saved as {output_wav}")

# from spleeter.separator import Separator

# Separar o áudio em dois componentes (voz e acompanhamento)
# separator = Separator('spleeter:2stems')
# separator.separate_to_file('ihsn5kZQEYA.wav', 'output')

import librosa
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa.display


# Carregar o arquivo de áudio
y, sr = librosa.load('ihsn5kZQEYA.wav')

# Reduzir o ruído
reduced_noise = nr.reduce_noise(y=y, sr=sr,thresh_n_mult_nonstationary=2, time_constant_s=0.2)

reduced_noise = nr.reduce_noise(y=reduced_noise, sr=sr, stationary=True)
# Filtrar valores não finitos
reduced_noise = np.nan_to_num(reduced_noise)

# Salvar o áudio com ruído reduzido
sf.write('ihsn5kZQEYA_reduced.wav', reduced_noise, sr)

# Carregar o arquivo de áudio
y, sr = librosa.load('ihsn5kZQEYA.wav')

# Criar os gráficos de som
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Com Ruído')

plt.subplot(2, 1, 2)
librosa.display.waveshow(reduced_noise, sr=sr)
plt.title('Sem Ruído')

plt.tight_layout()

# Salvar os gráficos em um arquivo PNG
plt.savefig('comparacao_ruido_ns.png')