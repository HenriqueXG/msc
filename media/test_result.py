# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import json
import pandas
from scipy import signal
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

if config['dataset'] == 'indoor':
    n_classes = 67
else:
    n_classes = 397

path_r = os.path.join(config['path'], 'media', f"test_result_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_1_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
with open(path_r, 'rb') as fp:
    classes_predicted = pickle.load(fp)

path_test = os.path.join(config['path'], 'data', f"test_{config['dataset']}_declarative.pkl")
if not os.path.exists(path_test):
    path_test = os.path.join(config['path'], 'data', "test_sun397_declarative_{:0>2d}.pkl".format(config['sun397_it']))
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
report = metrics.classification_report(test_data['Y'], classes_predicted['classes'], digits=3, output_dict=True)
df = pandas.DataFrame(report).transpose()

path_table = os.path.join(config['path'], 'media', f"pr_{config['dataset']}.tex")
with open(path_table, 'w') as fp:
    fp.write(df.to_latex())

precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(np.array(classes_predicted['y_test'])[:, i], np.array(classes_predicted['predictions'])[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall curves")

plt.savefig(os.path.join(config['path'], 'media', 'images', f"pr_{config['dataset']}.png"), dpi=300)
plt.show()

precision, recall, _ = precision_recall_curve(np.array(classes_predicted['y_test']).ravel(), np.array(classes_predicted['predictions']).ravel())
average_precision = average_precision_score(np.array(classes_predicted['y_test']), np.array(classes_predicted['predictions']), average="micro")

plt.plot(recall, precision, 'k--',
         label='micro-average Precision-Recall curve (area = {0:0.2f})'
               ''.format(average_precision))
plt.fill_between(recall, precision, color='#539ecd')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall micro-average curve")
plt.legend(loc="lower left")

plt.savefig(os.path.join(config['path'], 'media', 'images', f"pr_avg_{config['dataset']}.png"), dpi=300)
plt.show()