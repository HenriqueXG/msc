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

path_r = os.path.join(config['path'], 'media', f"{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}.pkl")
with open(path_r, 'rb') as fp:
    results = pickle.load(fp)

fig, ax = plt.subplots()
mean = [0] * len(results[0][:,1])
for r in results:
    ax.plot(r[:,0], r[:,1], color = 'black', lw = 1)
    mean = np.array(mean) + np.array(r[:,1])

ax.plot(results[0][:,0], mean/len(results), color = 'red', lw = 2)

ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')

plt.savefig(os.path.join(config['path'], 'media', 'images', 'alpha.png'), dpi=300)
plt.show()
