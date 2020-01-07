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

alphas = ['0.0']
datasets = [config['dataset']]
kernels = ['rbf', 'sigmoid', 'linear']
hidden_units = [100, 500, 1000, 2000]
activations = ['relu', 'logistic', 'tanh']
optimizers = ['adam', 'sgd']

for alpha in alphas:
    for dataset in datasets:
        for kernel in kernels:
            for hu in hidden_units:
                for activation in activations:
                    for optimizer in optimizers:
                        path_r = path_r = os.path.join(config['path'], 'media', f"exp_b_{alpha}_{dataset}_{kernel}_{hu}_{activation}_{optimizer}.pkl")
                        if os.path.exists(path_r):
                            with open(path_r, 'rb') as fp:
                                results = pickle.load(fp)
                            print(f"------ Result of: {alpha}_{dataset}_{kernel}_{hu}_{activation}_{optimizer} ------")
                            mean = sum(results)/len(results)
                            print(f"Mean: {mean}")
                            print('#####################')

path_r = os.path.join(config['path'], 'media', f"exp_b_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
with open(path_r, 'rb') as fp:
    results = pickle.load(fp)

fig, ax = plt.subplots()

x = np.arange(len(results))
ax.bar(x, results)

ax.set_ylim(0.6,1.0)
ax.set_xlabel('Runs')
ax.set_ylabel('Accuracy')
ax.set_xticks(x, (i+1 for i in range(len(results))))

plt.savefig(os.path.join(config['path'], 'media', 'images', 'exp_b.png'), dpi=300)
plt.show()
