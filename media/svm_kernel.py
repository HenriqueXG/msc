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

svmrbf = []
path = os.path.join(config['path'], 'media', 'svmrbf.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        svmrbf.append(float(line))

# svmpoly = []
# path = os.path.join(config['path'], 'media', 'svmpoly.txt')
# with open(path, 'r', encoding='ISO-8859-1') as archive:
#     for line in archive:
#         svmpoly.append(float(line))

svmlinear = []
path = os.path.join(config['path'], 'media', 'svmlinear.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        svmlinear.append(float(line))

svmsigmoid = []
path = os.path.join(config['path'], 'media', 'svmsigmoid.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        svmsigmoid.append(float(line))

fig, ax = plt.subplots()
ax.boxplot([svmrbf, svmlinear, svmsigmoid], showmeans=True)

ax.set_xlabel('Kernel')
ax.set_ylabel('Score')
ax.set_xticklabels(['RBF', 'Linear', 'Sigmoid'])
ax.grid(True)

plt.show()