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
    from lib.graph import Graph
    from lib.declarative import Declarative
except ImportError:
    from src.graph import Graph
    from src.declarative import Declarative

def open_indoor_test(declarative):
    # Open tess MIT Indoor 67 files

    path_test = os.path.join(config['path'], 'data', 'TestImages.txt')
    path_train = os.path.join(config['path'], 'data', 'TrainImages.txt')
    length = len(open(path_test).readlines())

    test_indoor_path = os.path.join(config['path'], 'data', 'test_indoor.pkl')
    test_data = None
    if os.path.exists(test_indoor_path): # Check if file exists
        with open(test_indoor_path, 'rb') as fp:
            test_data = pickle.load(fp)
    else:
        X_test = []
        Y_test = []

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for line in archive:
                path = os.path.join(config['path'], 'data', 'MITImages', line.strip())
                img = Image.open(path)
                img = declarative.img_channels(img)

                scene_class = Path(line).parts[0].strip() # Get name supervision from path
                vec = declarative.img2vec.get_vec(img) # Get vec from image

                X_test.append(vec)
                Y_test.append(scene_class)
        
        test_data = {'X':X_test, 'Y':Y_test}
        with open(test_indoor_path, 'wb') as fp:
            pickle.dump(test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    train_indoor_path = os.path.join(config['path'], 'data', 'train_indoor.pkl')
    train_data = None
    if os.path.exists(train_indoor_path): # Check if file exists
        with open(train_indoor_path, 'rb') as fp:
            train_data = pickle.load(fp)
    else:
        X_train = []
        Y_train = []

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for line in archive:
                path = os.path.join(config['path'], 'data', 'MITImages', line.strip())
                img = Image.open(path)
                img = declarative.img_channels(img)

                scene_class = Path(line).parts[0].strip() # Get name supervision from path
                vec = declarative.img2vec.get_vec(img) # Get vec from image

                X_train.append(vec)
                Y_train.append(scene_class)
        
        train_data = {'X':X_train, 'Y':Y_train}
        with open(test_indoor_path, 'wb') as fp:
            pickle.dump(train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return train_data, test_data, length

def test_indoor_svm(svm, train_data, test_data):
    # SVM test on MIT Indoor 67
    print('Testing SVM...')

    pred = {}
    svm_param = {}

    svm_param['model'] = 'svm'

    for idx in range(len(test_data['X'])):
        pred_ = svm.predict_proba([test_data['X'][idx]])[0]
        pred[idx] = pred_
    svm_param['classes'] = svm.classes_

    acc = svm.score(test_data['X'], test_data['Y'])
    svm_param['acc'] = acc

    return pred, svm_param

def test_indoor_graph(pam, train_data, test_data):
    # Graph test on MIT Indoor 67
    print('Testing graph...')

    path_test = os.path.join(config['path'], 'data', 'TestImages.txt')
    test_indoor_path_graph = os.path.join(config['path'], 'data', 'test_indoor_graph.pkl')

    if not os.path.exists(test_indoor_path_graph):        
        length = len(open(path_test).readlines())
        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get class supervision from path

                    path = os.path.join(config['path'], 'data', 'MITImages', line.strip())

                    im_fname = utils.download('', path = path)
                    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

                    box_ids, sscores, bboxes = pam.net(x)

                    boxes = []
                    scores = []
                    ids = []
                    for i in range(len(sscores[0])):
                        if sscores[0][i][0] > float(config['pam_threshold']): # Choose only detections that have confident higher than a threshold
                            boxes.append(bboxes.asnumpy()[0][i])
                            scores.append(float(sscores.asnumpy()[0][i][0]))
                            ids.append(int(box_ids.asnumpy()[0][i][0]))

                    ids = [*set(ids)]
                    ids_bool = [0] * len(pam.coco_classes)

                    for i in ids:
                        ids_bool[i] = 1

                    test_data['X'][idx] += ids_bool
                    
                except:
                    print('Error at {}'.format(line))
                    return

        print(len(test_data['X'][0]))
        with open(test_indoor_path_graph, 'wb') as fp:
            pickle.dump(test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:
        with open(test_indoor_path_graph, 'rb') as fp:
            test_data = pickle.load(fp)

    return test_data

def test_indoor(svm, declarative, train_data, test_data, graph_r, length):
    # Testing - Scenes Vectors - MIT Indoor 67
    acc = []

    svm_r, svm_param = test_indoor_svm(svm, train_data, test_data)

    for idx in range(len(test_data['X'])):
        v1 = svm_r[idx]
        svm_r[idx] = [(x-min(v1))/(max(v1)-min(v1)) for x in v1] # MinMax Normalization
    print(svm_param['acc'])
    corr = 0.0
    for idx, pred_ in svm_r.items():
        max_i = 0
        max_v = 0
        for i in range(len(pred_)):
            weight = graph_r['pred_weights'][idx][svm_param['classes'][i]]
            if pred_[i] > max_v and weight != 0.0:
                max_v = pred_[i]
                max_i = i
        pred_class = svm_param['classes'][max_i]
        scene_class = test_data['Y'][idx]

        if pred_class == scene_class:
            corr += 1.0

    return corr/length

if __name__ == '__main__':
    declarative = Declarative()

    train_data, test_data, length = open_indoor_test(declarative)
    
    svm = SVC(kernel='rbf', probability=True).fit(train_data['X'], train_data['Y'])

    pam = Graph()
    graph_r = test_indoor_graph(pam)

    r = test_indoor(svm, declarative, train_data, test_data, graph_r, length)

    print(r)