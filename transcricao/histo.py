import matplotlib.pyplot as plt
import numpy as np

# Define o tamanho do vetor
tamanho = 500

# Gera valores aleatórios e adiciona alguns picos
np.random.seed(0)  # For reproducible results
valores = np.random.uniform(0, 0.6, size=tamanho)

# Adiciona picos
i = 0
for i in range(350,355):
    if i % 2 == 0:
        valores[i] = 0.8
    else:
        valores[i] = 0.95
    i += 1

# Cria um vetor x para o gráfico
x = np.linspace(0, 70, tamanho)

# Cria o gráfico com traços sem marcadores
plt.figure(figsize=(12, 6))
plt.plot(x, valores, linestyle='-', color='green')

# Adiciona títulos e rótulos aos eixos
# plt.title('Histograma Fake com Traços')
plt.xlabel('Relevância')
plt.ylabel('Segundos')

# Define os limites do eixo x e y
plt.xlim(0, 70)  # Define o intervalo do eixo x
plt.ylim(0, 1)  # Limita o eixo y

# Salva a imagem
plt.savefig('/home/luisa/Documents/identificacao-momentos-relevantes/transcricao/histograma_fake.png')

# Fecha a figura para liberar memória
plt.close()
