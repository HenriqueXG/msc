# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import json

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

nntanh = []
path = os.path.join(config['path'], 'media', 'nntanh.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        nntanh.append(float(line))

nnrelu = []
path = os.path.join(config['path'], 'media', 'nnrelu.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        nnrelu.append(float(line))

nnlogistic = []
path = os.path.join(config['path'], 'media', 'nnlogistic.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        nnlogistic.append(float(line))

fig, ax = plt.subplots()
ax.boxplot([nntanh, nnrelu, nnlogistic], showmeans=True)

ax.set_xlabel('Activation functions')
ax.set_ylabel('Score')
ax.set_xticklabels(['Tanh', 'Relu', 'Sigmoid'])
ax.grid(True)

plt.show()