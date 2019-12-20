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

path_r = os.path.join(config['path'], 'media', f"exp_b_{config['alpha']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}.pkl")
with open(path_r, 'rb') as fp:
    results = pickle.load(fp)

mean = sum(results)/len(results)
print(f"Mean: {mean}")

fig, ax = plt.subplots()

x = np.arange(len(results))
ax.bar(x, results)

ax.set_ylim(0.6,1.0)
ax.set_xlabel('Runs')
ax.set_ylabel('Accuracy')
ax.set_xticks(x, (i+1 for i in range(len(results))))

plt.savefig(os.path.join(config['path'], 'media', 'images', 'exp_b.png'), dpi=300)
plt.show()
