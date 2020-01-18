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

path_r = os.path.join(config['path'], 'media', f"test_result_{config['pam_threshold']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
with open(path_r, 'rb') as fp:
    classes_predicted = pickle.load(fp)

print(classes_predicted)

fig, ax = plt.subplots()



ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')

plt.savefig(os.path.join(config['path'], 'media', 'images', 'pr.png'), dpi=300)
plt.show()
