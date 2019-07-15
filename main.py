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

def test_indoor_model(svm, nn, train_data, test_data):
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

def test_indoor(svm, nn, train_data, test_data):
    # Testing - MIT Indoor 67
    pred_svm, pred_nn, param = test_indoor_model(svm, nn, train_data, test_data)
    # return param['svm_scr']

    r = []
    for alpha in np.arange(0.0, 1.05, 0.05):
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

        r.append([alpha, corr/len(test_data['Y'])])
    return np.array(r)

if __name__ == '__main__':
    if os.path.exists(os.path.join(config['path'], 'data', 'test_indoor_spatial.pkl')) and os.path.exists(os.path.join(config['path'], 'data', 'train_indoor_spatial.pkl')):
        # If Spatial Memory is preprocessed, it has all information preprocessed by the other memories. Thus, the others ones don't have to be loaded. 
        spatial = Spatial(None, None)
    else:
        declarative = Declarative()
        pam = PAM(declarative.train_data, declarative.test_data)
        spatial = Spatial(pam.train_data, pam.test_data)

    print('Vector dimension: {}'.format(len(spatial.train_data['X'][0])))

    results = []
    for i in range(config['it']):
        sys.stdout.write(f"Fitting/Testing... {i+1}/{config['it']}\r")
        svm = SVC(kernel=config['kernel'], probability=True).fit(spatial.train_data['X'], spatial.train_data['Y'])
        # nn = None
        nn = MLPClassifier(hidden_layer_sizes=(config['hidden_units'],), activation=config['activation']).fit(spatial.train_data['X'], spatial.train_data['Y'])

        r = test_indoor(svm, nn, spatial.train_data, spatial.test_data)
        results.append(r)
        # path_r = os.path.join(config['path'], 'media', 'svmsigmoid.txt')
        # with open(path_r, 'a') as fp:
        #     fp.write(str(r) + '\n') 
        
    path_r = os.path.join(config['path'], 'media', f"{config['kernel']}_{config['hidden_units']}_{config['activation']}.pkl")
    with open(path_r, 'wb') as fp:
        pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL) 

