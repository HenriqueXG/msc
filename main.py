# Main - Workspace / Global Workspace
## Henrique X. Goulart

import json
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import spatial, interpolate
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from gluoncv import data, utils

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'])

try:
    from lib.pam import PAM
    from lib.declarative import Declarative
    from lib.spatial import Spatial
except ImportError:
    from src.pam import PAM
    from src.declarative import Declarative
    from src.spatial import Spatial

def test_indoor_model(svm, nn, test_data):
    # Perform test on MIT Indoor 67
    pred_svm = {}
    pred_nn = {}
    param = {}

    for idx in range(len(test_data['X'])):
        pred_svm[idx] = svm.predict_proba([test_data['X'][idx]])[0]
        pred_nn[idx] = nn.predict_proba([test_data['X'][idx]])[0]

    param['classes'] = svm.classes_
    param['nn_scr'] = nn.score(test_data['X'], test_data['Y'])

    corr = 0.0
    for idx in range(len(test_data['X'])):
        max_i = 0
        max_v = 0
        pred_ = pred_svm[idx]

        for i in range(len(pred_)):
            if pred_[i] > max_v:
                max_v = pred_[i]
                max_i = i
        pred_class = param['classes'][max_i]
        scene_class = test_data['Y'][idx]

        if pred_class == scene_class:
            corr += 1.0
    param['svm_scr'] = corr/len(test_data['X'])

    return pred_svm, pred_nn, param

def test_indoor(svm, nn, test_data, alpha):
    # Testing - MIT Indoor 67
    pred_svm, pred_nn, param = test_indoor_model(svm, nn, test_data)

    predictions = []

    for idx in range(len(test_data['X'])):
        v1 = pred_svm[idx]
        v2 = pred_nn[idx]

        # MinMax Normalization
        v1 = [(x-min(v1))/(max(v1)-min(v1)) for x in v1]
        v2 = [(x-min(v2))/(max(v2)-min(v2)) for x in v2]

        pred_ = [v1[i]*alpha + v2[i]*(1.0 - alpha) for i in range(len(v1))]
        predictions.append(pred_)

    corr = 0.0
    for idx, pred_ in enumerate(predictions):
        max_i = 0
        max_v = 0
        for i in range(len(pred_)):
            if pred_[i] > max_v:
                max_v = pred_[i]
                max_i = i
        pred_class = param['classes'][max_i]
        scene_class = test_data['Y'][idx]

        if pred_class == scene_class:
            corr += 1.0

    return corr/len(test_data['Y'])

def CSM(pam, spatial, declarative):
    train_data = {'X':[], 'Y':[]}
    test_data = {'X':[], 'Y':[]}

    for sample_a, sample_b, sample_c in zip(pam.train_data['X'], spatial.train_data['X'], declarative.train_data['X']):
        sample = np.concatenate((np.array(sample_a), np.array(sample_b), np.array(sample_c)), axis=0).tolist()
        train_data['X'].append(sample)

    train_data['Y'] = declarative.train_data['Y']

    for sample_a, sample_b, sample_c in zip(pam.test_data['X'], spatial.test_data['X'], declarative.test_data['X']):
        sample = np.concatenate((np.array(sample_a), np.array(sample_b), np.array(sample_c)), axis=0).tolist()
        test_data['X'].append(sample)

    test_data['Y'] = declarative.test_data['Y']
    
    return train_data, test_data

def exp_a(train_data, test_data):
    results = []
    for i in range(config['it']):
        sys.stdout.write(f"Fitting and Testing... {i+1}/{config['it']}\n")

        svm = SVC(kernel=config['kernel'], probability=True).fit(train_data['X'], train_data['Y'])
        nn = MLPClassifier(hidden_layer_sizes=(config['hidden_units'],), activation=config['activation']).fit(train_data['X'], train_data['Y'])

        r = []
        for alpha in np.arange(0.0, 1.05, 0.05):
            r.append([alpha, test_indoor(svm, nn, test_data, alpha)])
        results.append(np.array(r))
        
    path_result = os.path.join(config['path'], 'media', f"{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}.pkl")
    with open(path_result, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    pam = PAM()
    spatial = Spatial()
    declarative = Declarative()

    train_data, test_data = CSM(pam, spatial, declarative)

    print('CSM dimension: {}'.format(len(train_data['X'][0])))

    exp_a(train_data, test_data)

