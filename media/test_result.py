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

path_test = os.path.join(config['path'], 'data', 'test_indoor_declarative.pkl')
with open(path_test, 'rb') as fp:
    test_data = pickle.load(fp)

corr = 0
for pred_class, scene_class in zip(classes_predicted['classes'], test_data['Y']):
    if pred_class == scene_class:
        corr += 1
print(corr/len(test_data['Y']))

fig, ax = plt.subplots()



ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')

plt.savefig(os.path.join(config['path'], 'media', 'images', 'pr.png'), dpi=300)
plt.show()
