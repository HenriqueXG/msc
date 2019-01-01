# Main - Workspace
## Henrique X. Goulart

import json
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image
from pathlib import Path
from scipy import spatial
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from gluoncv import data, utils

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'])

try:
    from lib.pam import PAM
    from lib.declarative import Declarative
except ImportError:
    from src.pam import PAM
    from src.declarative import Declarative

def test_indoor_svm(svm, train_data, test_data):
    # SVM test on MIT Indoor 67
    print('Testing SVM...')

    svm_param = {}

    svm_param['classes'] = svm.classes_

    acc = svm.score(test_data['X'], test_data['Y'])
    svm_param['acc'] = acc

    return svm_param

def test_indoor(svm, declarative, train_data, test_data):
    # Testing - Scenes Vectors - MIT Indoor 67
    svm_param = test_indoor_svm(svm, train_data, test_data)

    return svm_param['acc']

if __name__ == '__main__':
    declarative = Declarative()

    pam = PAM(declarative.train_data, declarative.test_data)
    
    print('Vector dimension: {}'.format(len(pam.train_data['X'][0])))

    svm = SVC(kernel='rbf').fit(pam.train_data['X'], pam.train_data['Y'])
    r = test_indoor(svm, declarative, pam.train_data, pam.test_data)

    print(r)