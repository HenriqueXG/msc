# Main - Workspace / Global Workspace
## Henrique X. Goulart

import json
import sys
import os
import pickle
import datetime
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
except (ImportError, ModuleNotFoundError) as e:
    from src.pam import PAM
    from src.declarative import Declarative
    from src.spatial import Spatial

def model_pred(svm, nn, test_data):
    # Predictions
    pred_svm = {}
    pred_nn = {}
    param = {}

    for idx in range(len(test_data['X'])):
        pred_svm[idx] = svm.predict_proba([test_data['X'][idx]])[0]
        pred_nn[idx] = nn.predict_proba([test_data['X'][idx]])[0]

    param['classes'] = svm.classes_

    return pred_svm, pred_nn, param

def test(svm, nn, test_data, alpha):
    # Testing predictions
    pred_svm, pred_nn, param = model_pred(svm, nn, test_data)

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
    classes = []
    y_test = []
    for idx, pred_ in enumerate(predictions):
        max_i = 0
        max_v = 0
        for i in range(len(pred_)):
            if pred_[i] > max_v:
                max_v = pred_[i]
                max_i = i
        pred_class = param['classes'][max_i]
        scene_class = test_data['Y'][idx]
        classes.append(pred_class)
        
        if config['dataset'] == 'indoor':
            y = list([0] * 67)
        elif config['dataset'] == 'sun397':
            y = list([0] * 397)
        test_idx = param['classes'].tolist().index(scene_class)
        y[test_idx] = 1
        y_test.append(y)

        if pred_class == scene_class:
            corr += 1.0
    
    classes_predicted = {'classes': classes, 'predictions': predictions, 'y_test': y_test}
    path_result = os.path.join(config['path'], 'media', f"test_result_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
    with open(path_result, 'wb') as fp:
        pickle.dump(classes_predicted, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    print("Experiment A - Running...")
    results = []
    for i in range(config['it']):
        sys.stdout.write(f"Fitting and Testing... {i+1}/{config['it']} -- {datetime.datetime.now()}\n")

        svm = SVC(kernel=config['kernel'], probability=True, gamma='scale').fit(train_data['X'], train_data['Y'])
        nn = MLPClassifier(hidden_layer_sizes=(config['hidden_units'],), activation=config['activation'], solver=config['optimizer']).fit(train_data['X'], train_data['Y'])

        r = []
        for alpha in np.arange(0.0, 1.05, 0.05):
            r.append([alpha, test(svm, nn, test_data, alpha)])
        results.append(np.array(r))
        
    path_result = os.path.join(config['path'], 'media', f"exp_a_{config['pam_threshold']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
    with open(path_result, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Experiment A - Done! -- {datetime.datetime.now()}")

def exp_b(train_data, test_data):
    print("Experiment B - Running...")
    results = []
    for i in range(config['it']):
        sys.stdout.write(f"Fitting and Testing... {i+1}/{config['it']} -- {datetime.datetime.now()}\n")

        svm = SVC(kernel=config['kernel'], probability=True, gamma='scale').fit(train_data['X'], train_data['Y'])
        nn = MLPClassifier(hidden_layer_sizes=(config['hidden_units'],), activation=config['activation'], solver=config['optimizer']).fit(train_data['X'], train_data['Y'])

        r = test(svm, nn, test_data, config['alpha'])
        results.append(r)
        
    path_result = os.path.join(config['path'], 'media', f"exp_b_{config['pam_threshold']}_{config['alpha']}_{config['dataset']}_{config['kernel']}_{config['hidden_units']}_{config['activation']}_{config['optimizer']}.pkl")
    with open(path_result, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Experiment B - Done! -- {datetime.datetime.now()}")

if __name__ == '__main__':
    pam = PAM()
    spatial = Spatial()
    declarative = Declarative()

    train_data, test_data = CSM(pam, spatial, declarative)
    del pam
    del spatial
    del declarative

    print('CSM dimension: {}'.format(len(train_data['X'][0])))

    if config['exp_a']:
        exp_a(train_data, test_data)
    if config['exp_b']:
        exp_b(train_data, test_data)

