# Main - Workspace
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
    print('Testing...')

    pred_svm = {}
    pred_nn = {}
    param = {}

    for idx in range(len(test_data['X'])):
        pred_svm[idx] = svm.predict_proba([test_data['X'][idx]])[0]
        pred_nn[idx] = nn.predict_proba([test_data['X'][idx]])[0]

    param['classes'] = svm.classes_
    param['svm_scr'] = svm.score(test_data['X'], test_data['Y'])
    param['nn_scr'] = nn.score(test_data['X'], test_data['Y'])

    return pred_svm, pred_nn, param

def test_indoor(svm, nn, declarative, train_data, test_data):
    # Testing - MIT Indoor 67
    pred_svm, pred_nn, param = test_indoor_model(svm, nn, train_data, test_data)

    r = []
    for alpha in np.linspace(0.0, 1.0, 10):
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
    return np.array(r), param['svm_scr'], param['nn_scr']

if __name__ == '__main__':
    declarative = Declarative()
    pam = PAM(declarative.train_data, declarative.test_data)
    spatial = Spatial(pam.train_data, pam.test_data)

    print('Vector dimension: {}'.format(len(spatial.train_data['X'][0])))

    print('Fitting...')
    svm = SVC(kernel='rbf', probability=True).fit(spatial.train_data['X'], spatial.train_data['Y'])
    nn = MLPClassifier(hidden_layer_sizes=(1000, )).fit(spatial.train_data['X'], spatial.train_data['Y'])

    r, svm_scr, nn_scr = test_indoor(svm, nn, declarative, spatial.train_data, spatial.test_data)

    # X = np.array(r[:,0]).reshape((10,10))
    # Y = np.array(r[:,1]).reshape((10,10))
    # Z = np.array(r[:,2]).reshape((10,10))

    # plt.contourf(X,Y,Z)
    # plt.colorbar()

    plt.plot(r[:,0], r[:,1])

    plt.show()
    
    path_r = os.path.join(config['path'], 'media', 'rbf_1000.txt')
    with open(path_r, 'a') as fp:
        fp.write(str(svm_scr) + ',' + str(nn_scr))    
