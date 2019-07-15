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

markers = ['x','>','+','*','D']

hidden_units = 1000
activation = 'logistic'
kernel = 'rbf'

path_r = os.path.join(config['path'], 'media', f'{kernel}_{hidden_units}_{activation}.pkl')
with open(path_r, 'rb') as fp:
    results = pickle.load(fp)

fig, ax = plt.subplots()
for r, m in zip(results, markers):
    print(r[:,0])
    ax.plot(r[:,0], r[:,1], marker = m, lw = 1)

# X = np.array(r[:,0]).reshape((10,10))
# Y = np.array(r[:,1]).reshape((10,10))
# Z = np.array(r[:,2]).reshape((10,10))

# plt.contourf(X,Y,Z)
# plt.colorbar()

ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')

plt.savefig(os.path.join(config['path'], 'media', 'images', 'alpha.png'), dpi=300)
plt.show()