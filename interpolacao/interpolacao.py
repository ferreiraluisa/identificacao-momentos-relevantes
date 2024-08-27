import numpy as np
import json
import matplotlib.pyplot as plt

# Carregando os dados dos arquivos JSON
with open('detection.json', 'r') as f:
    dados_detection = json.load(f)

hist_people = dados_detection['people']
hist_gun = dados_detection['guns']

with open('pose.json', 'r') as f:
    dados_pose = json.load(f)

hist_pose = dados_pose['people']

with open('sons.json', 'r') as f:
    dados_sons = json.load(f)

hist_sons = dados_sons['guns']

with open('transcricao.json', 'r') as f:
    dados_transcricao = json.load(f)

hist_trans = dados_transcricao['hist']

# Função de interpolação
def interpolacao(hist, n):
    hist = np.array(hist)
    x = np.linspace(0, len(hist)-1, n)
    return np.interp(x, np.arange(len(hist)), hist)

# Interpolando hist_sons e hist_trans para o mesmo tamanho de hist_people
n = len(hist_people)
hist_sons = interpolacao(hist_sons, n)
hist_trans = interpolacao(hist_trans, n)

# Verificando o tamanho dos histogramas
# print(len(hist_sons)) 
print(len(hist_people))
print(len(hist_gun))
print(len(hist_pose))
print(len(hist_trans))

# Calculando o histograma real usando o valor máximo de cada histograma interpolado
hist_real = [max(hist_people[i], hist_gun[i], hist_pose[i], hist_trans[i]) for i in range(n)]
hist_real = np.array(hist_real)
hist_real = hist_real.astype(float)
print(hist_real)

num_frames = len(hist_real)

# Calculando o tempo em segundos para cada frame
fps = 30
time_seconds = np.arange(num_frames) / fps

# Plotando o histograma com o eixo x em segundos
plt.bar(time_seconds, hist_real, width=0.03, edgecolor='black')  # Ajuste de largura para melhorar a visualização

# Configurando os eixos
plt.xlabel('Tempo (segundos)')
plt.ylabel('Valores normalizados (0 a 1)')
plt.ylim(0, 1)  # Garantindo que o eixo Y vá de 0 a 1

# Mostrar o gráfico
plt.savefig('histograma.png')



