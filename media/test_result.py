# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import json
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve

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

# Print the confusion matrix
print(metrics.confusion_matrix(test_data['Y'], classes_predicted['classes']))

# Print the precision and recall, among other metrics
print(metrics.classification_report(test_data['Y'], classes_predicted['classes'], digits=3))
print(classes_predicted['y_test'])
fig, ax = plt.subplots()

precision = dict()
recall = dict()
for i in range(67):
    precision[i], recall[i], _ = precision_recall_curve(classes_predicted['y_test'][:, i], classes_predicted['predictions'][:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')

plt.savefig(os.path.join(config['path'], 'media', 'images', 'pr.png'), dpi=300)
plt.show()
