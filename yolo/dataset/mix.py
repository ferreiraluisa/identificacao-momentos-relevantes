import os
import shutil
import sys

origem_x = sys.argv[1]
origem_y = sys.argv[2]
destino = sys.argv[3]


if not os.path.exists(destino):
    os.makedirs(destino)

for arquivo_x in os.listdir(origem_x):
    shutil.move(os.path.join(origem_x, arquivo_x), destino)

for arquivo_y in os.listdir(origem_y):
    shutil.move(os.path.join(origem_y, arquivo_y), destino)


