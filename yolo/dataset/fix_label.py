import os
import sys

path = sys.argv[1]

files = os.listdir(path)

for f in files:
    with open(path + f, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for linha in lines:
        elementos = linha.split()
        elementos[0] = '80'
        new = ' '.join(elementos)
        new_lines.append(new)
    
    with open(path + f, 'w') as file:
        file.writelines(new_lines)

