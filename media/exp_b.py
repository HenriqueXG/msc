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

pam_thresholds = ['0.5', '0.7', '0.9']
alphas = ['0.0', '0.5', '1.0']
datasets = [config['dataset']]
kernels = ['rbf', 'sigmoid', 'linear']
hidden_units = [100, 500, 1000, 2000]
activations = ['relu', 'logistic', 'tanh']
optimizers = ['adam', 'sgd']

for pam_threshold in pam_thresholds:
    for alpha in alphas:
        for dataset in datasets:
            for kernel in kernels:
                for hu in hidden_units:
                    for activation in activations:
                        for optimizer in optimizers:
                            path_r = path_r = os.path.join(config['path'], 'media', f"exp_b_{pam_threshold}_{alpha}_{dataset}_{kernel}_{hu}_{activation}_{optimizer}.pkl")
                            if os.path.exists(path_r):
                                with open(path_r, 'rb') as fp:
                                    results = pickle.load(fp)
                                print(f"------ Result of: {pam_threshold}_{alpha}_{dataset}_{kernel}_{hu}_{activation}_{optimizer} ------")
                                mean = np.mean(results)
                                std = np.std(results)
                                print(f"Std: {std}")
                                print(f"Mean: {mean}")
                                print('#####################')
                                fig, ax = plt.subplots()

                                x = np.arange(len(results))
                                ax.bar(x, results)

                                ax.set_ylim(0.6,1.0)
                                ax.set_xlabel('Runs')
                                ax.set_ylabel('Accuracy')
                                ax.set_xticks(x, (i+1 for i in range(len(results))))

                                plt.draw()
                                fig.savefig(os.path.join(config['path'], 'media', 'images', f"exp_b_{pam_threshold}_{alpha}_{dataset}_{kernel}_{hu}_{activation}_{optimizer}.png".replace('.','_')), dpi=300)

if config['dataset'] == 'sun397':
    results = []
    for i in range(10):
        path_r = os.path.join(config['path'], 'media', f"exp_b_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_{i+1}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
        if os.path.exists(path_r):
            with open(path_r, 'rb') as fp:
                r = pickle.load(fp)
                results.append(r[0])
else:
    path_r = os.path.join(config['path'], 'media', f"B_exp_b_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
    with open(path_r, 'rb') as fp:
        results = pickle.load(fp)

mean = np.mean(results)
std = np.std(results)
print(f"Std: {std}")
print(f"Mean: {mean}")

plt.clf()
fig, ax = plt.subplots()

x = np.arange(len(results))
ax.bar(x, results)

ax.set_ylim(0.6,1.0)
ax.set_xlabel('Runs')
ax.set_ylabel('Accuracy')
ax.set_xticks(x, (i+1 for i in range(len(results))))

plt.savefig(os.path.join(config['path'], 'media', 'images', 'exp_b.png'), dpi=300)
