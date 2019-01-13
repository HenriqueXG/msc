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

r1000 = []
path = os.path.join(config['path'], 'media', 'nn1000.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        r1000.append(float(line))

r500 = []
path = os.path.join(config['path'], 'media', 'nn500.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        r500.append(float(line))

r100 = []
path = os.path.join(config['path'], 'media', 'nn100.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        r100.append(float(line))

r50 = []
path = os.path.join(config['path'], 'media', 'nn50.txt')
with open(path, 'r', encoding='ISO-8859-1') as archive:
    for line in archive:
        r50.append(float(line))

fig, ax = plt.subplots()
ax.boxplot([r50, r100, r500, r1000], showmeans=True)

ax.set_xlabel('Number of hidden units')
ax.set_ylabel('Score')
ax.set_xticklabels(['50', '100', '500', '1000'])
ax.grid(True)

plt.show()