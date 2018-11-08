## Henrique X. Goulart

import json
import sys
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image
from pathlib import Path
from scipy import spatial
from sklearn.svm import SVC

random.seed(42)

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

def open_indoor_test():
    # Open tess MIT Indoor 67 files

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')
    path_train = os.path.join(declarative.config['path'], 'data', 'TrainImages.txt')
    length = len(open(path_test).readlines())

    test_indoor_path = os.path.join(declarative.config['path'], 'data', 'test_indoor.pkl')
    test_data = None
    if os.path.exists(test_indoor_path): # Check if file exists
        with open(test_indoor_path, 'rb') as fp:
            test_data = pickle.load(fp)
    else:
        X_test = []
        Y_test = []

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for line in archive:
                path = os.path.join(declarative.config['path'], 'data', 'MITImages', line.strip())
                img = Image.open(path)
                img = declarative.img_channels(img)

                scene_class = Path(line).parts[0].strip() # Get name supervision from path
                vec = declarative.img2vec.get_vec(img) # Get vec from image

                X_test.append(vec)
                Y_test.append(scene_class)
        
        test_data = {'X':X_test, 'Y':Y_test}
        with open(test_indoor_path, 'wb') as fp:
            pickle.dump(test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    train_indoor_path = os.path.join(declarative.config['path'], 'data', 'train_indoor.pkl')
    train_data = None
    if os.path.exists(train_indoor_path): # Check if file exists
        with open(train_indoor_path, 'rb') as fp:
            train_data = pickle.load(fp)
    else:
        X_train = []
        Y_train = []

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for line in archive:
                path = os.path.join(declarative.config['path'], 'data', 'MITImages', line.strip())
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

def test_indoor_svm(train_data, test_data, output):
    # SVM test on MIT Indoor 67
    print('Testing SVM...')

    svm = SVC(kernel='rbf', probability=True).fit(train_data['X'],train_data['Y'])

    pred = svm.predict_proba(test_data['X'])
    # acc = svm.score(test_data['X'], test_data['Y'])

    output.put(pred)

def test_indoor_graph(declarative, graph, test_data, length, output):
    # Graph test on MIT Indoor 67

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')

    corr = 0.0

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, line in enumerate(archive):
            sys.stdout.write('Testing graph... ' + str(idx+1) + '/' + str(length) + '\n')
            
            path = os.path.join(declarative.config['path'], 'data', 'MITImages', line.strip())
            img = Image.open(path)
            img = declarative.img_channels(img)

            if not test_data:
                scene_class = Path(line).parts[0].strip() # Get name supervision from path
            else:
                scene_class = test_data['Y'][idx]

            region_vectors = declarative.extract_regions(img)
                        
            processes_reg = []
            output_reg = mp.Queue()
            for reg in region_vectors:
                processes_reg.append(mp.Process(target=graph.get_obj, args=(reg, output_reg)))
            for p in processes_reg:
                p.start()
            for p in processes_reg:
                p.join()
            objects = [*set([output_reg.get() for p in processes_reg])]

            neighbours = []
            for obj in objects:
                neighbours += graph.get_subgraph(obj)
            neighbours = [*set(neighbours)]

            distances = []
            for co_occurrence in declarative.declarative_data['co_occurrences']:
                if all(elem in co_occurrence['co_occurrence'] for elem in neighbours):
                    distance = np.inf
                    pred_class = ''

                    for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                        d = spatial.distance.cosine(co_occurrence['scene_vec'], class_vec)

                        if d < distance:
                            pred_class = class_name
                            distance = d
                    
                    distances.append({'d':distance, 'class_name':pred_class})

            distance = np.inf
            pred_class = ''
            for ds in distances:
                d = ds['d']
                class_name = ds['class_name']

                if d < distance:
                    pred_class = class_name
                    distance = d

            if pred_class == scene_class:
                corr += 1.0

    output.put(corr/length)

def test_indoor_cosine(declarative, test_data, length, output):
    # Cosine test on MIT Indoor 67
    print('Testing cosine')

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')

    corr = 0.0
    pred = []

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, _ in enumerate(archive):
            vec = test_data['X'][idx]
            scene_class = test_data['Y'][idx]

            distance = np.inf
            pred_class = ''
            _pred = []

            for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                d = 1.0 - spatial.distance.cosine(vec, class_vec)

                _pred.append(d)

                if d < distance:
                    pred_class = class_name
                    distance = d

            if pred_class == scene_class:
                corr += 1.0
            
            pred.append(_pred)
    
    output.put(pred)

if __name__ == '__main__':
    output = mp.Queue()

    # graph = Graph(arch = config['arch_obj'])
    declarative = Declarative()

    print('Testing - Scenes Vectors - MIT Indoor 67')
    train_data, test_data, length = open_indoor_test()

    processes = []
    processes.append(mp.Process(target=test_indoor_cosine, args=(declarative, test_data, length, output)))
    processes.append(mp.Process(target=test_indoor_svm, args=(train_data, test_data, output)))
    # processes.append(mp.Process(target=test_indoor_graph, args=(declarative, graph, test_data, length, output)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    results = [output.get() for p in processes]

    svm_r = results[0]
    cos_r = results[1]

    r = [np.array(svm_r[i])*0.9 + np.array(cos_r[i])*0.1 for i in range(len(svm_r))]

    # corr = 0.0
    # for idx, sample in enumerate(r):
    #     i = sample.index(max(sample))
    #     if test_data['Y'][idx] == 
    
    # print('Accuracy on MIT Indoor 67: {} / {}'.format(results[0], results[1]))
