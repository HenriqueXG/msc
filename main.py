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

def sim_classes(declarative, vectors):
    # Find eligible classes given a set of vectors
    eligible_classes = []

    for vec in vectors:
        distance = np.inf
        pred_class = ''

        for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
            d = spatial.distance.cosine(vec, class_vec)

            if d < distance:
                pred_class = class_name
                distance = d
        
        eligible_classes.append(pred_class)
    eligible_classes = [*set(eligible_classes)]


    return eligible_classes

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

def test_indoor_svm(svm, train_data, test_data, output):
    # SVM test on MIT Indoor 67
    print('Testing SVM...')

    pred = {}
    pred['model'] = 'svm'

    for idx in range(len(test_data['X'])):
        pred_ = svm.predict_proba([test_data['X'][idx]])[0]
        pred[idx] = pred_
    pred['classes'] = svm.classes_

    acc = svm.score(test_data['X'], test_data['Y'])
    pred['acc'] = acc

    output.put(pred)

def test_indoor_graph(declarative, test_data, length):
    # Graph test on MIT Indoor 67
    print('Testing graph...')

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')
    path_graph_indoor_dist = os.path.join(config['path'], 'data', 'graph_indoor_dist.pkl')

    corr = 0.0
    pred = {}
    pred['model'] = 'graph'
    pred['distances'] = []

    with open(path_graph_indoor_dist, 'rb') as fp:
        pred['distances'] = pickle.load(fp)

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, _ in enumerate(archive):
            scene_class = test_data['Y'][idx]

            distance = np.inf
            pred_class = ''
            for ds in pred['distances'][idx]:
                d = ds['d']
                class_name = ds['class_name']

                if d < distance:
                    pred_class = class_name
                    distance = d

            if pred_class == scene_class:
                corr += 1.0
    
    pred['acc'] = corr/length

    return pred

def test_indoor_graph_create(declarative, graph, test_data, length):
    # Graph test on MIT Indoor 67 and create pre-test data
    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')
    path_graph_indoor_dist = os.path.join(config['path'], 'data', 'graph_indoor_dist.pkl')

    corr = 0.0
    pred = {}
    pred['model'] = 'graph'
    pred['distances'] = []

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, line in enumerate(archive):            
            sys.stdout.write('Testing graph... ' + str(idx+1) + '/' + str(length) + '\r')
            
            scene_class = test_data['Y'][idx]

            path = os.path.join(config['path'], 'data', 'MITImages', line.strip())
            img = Image.open(path)
            img = declarative.img_channels(img)

            region_vectors = graph.extract_regions(img)
                        
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
                if all(elem in neighbours for elem in co_occurrence['co_occurrence']):
                    distance = np.inf
                    pred_class = ''

                    for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                        d = spatial.distance.cosine(co_occurrence['scene_vec'], class_vec)

                        if d < distance:
                            pred_class = class_name
                            distance = d
                    
                    distances.append({'d':distance, 'class_name':pred_class})

            pred['distances'].append(distances)

            distance = np.inf
            pred_class = ''
            for ds in pred['distances'][idx]:
                d = ds['d']
                class_name = ds['class_name']

                if d < distance:
                    pred_class = class_name
                    distance = d

            if pred_class == scene_class:
                corr += 1.0

    pred['acc'] = corr/length

    with open(path_graph_indoor_dist, 'wb') as fp:
        pickle.dump(pred['distances'], fp, protocol=pickle.HIGHEST_PROTOCOL)

    return pred

def test_indoor_cosine(declarative, test_data, length, output):
    # Cosine test on MIT Indoor 67
    print('Testing cosine...')

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')

    corr = 0.0
    pred = {}
    pred['model'] = 'cosine'

    classes_ = [class_name for class_name, _ in declarative.declarative_data['scene_vectors'].items()]
    pred['classes'] = classes_

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, _ in enumerate(archive):
            vec = np.array(test_data['X'][idx])
            scene_class = test_data['Y'][idx]

            distance = np.inf
            pred_class = ''

            pred_ = []

            for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                d = spatial.distance.cosine(vec, class_vec)

                pred_.append(1.0 - d)

                if d < distance:
                    pred_class = class_name
                    distance = d

            if pred_class == scene_class:
                corr += 1.0

            pred[idx] = pred_
    
    pred['acc'] = corr/length
        
    output.put(pred)

def test_indoor(svm, declarative, train_data, test_data, alphas):
    # Testing - Scenes Vectors - MIT Indoor 67
    acc = []

    # path_graph_indoor_eligible = os.path.join(config['path'], 'data', 'graph_indoor_eligible.pkl')
    # if os.path.exists(path_graph_indoor_eligible):
    #     gph_r = test_indoor_graph(declarative, test_data, length)
    # else:
    #     graph = Graph()
    #     gph_r = test_indoor_graph_create(declarative, graph, test_data, length)

    for alpha in alphas:
        output = mp.Queue()

        processes = []
        processes.append(mp.Process(target=test_indoor_cosine, args=(declarative, test_data, length, output)))
        processes.append(mp.Process(target=test_indoor_svm, args=(svm, train_data, test_data, output)))

        for p in processes:
            p.start()

        results = [output.get() for p in processes]

        for p in processes:
            p.join()

        for r in results:
            if r['model'] == 'svm':
                svm_r = r
            elif r['model'] == 'cosine':
                cos_r = r
        
        pred = []

        for idx in range(len(test_data['X'])):
            v1 = svm_r[idx]
            v2 = cos_r[idx]

            # MinMax Normalization
            v1 = [(x-min(v1))/(max(v1)-min(v1)) for x in v1]
            v2 = [(x-min(v2))/(max(v2)-min(v2)) for x in v2]

            pred_ = [v1[i]*alpha + v2[i]*(1.0 - alpha) for i in range(len(v1))]
            pred.append(pred_)

        corr = 0.0
        for idx, pred_ in enumerate(pred):
            max_i = 0
            max_v = 0
            for i in range(len(pred_)):
                if pred_[i] > max_v:
                    max_v = pred_[i]
                    max_i = i
            pred_class = svm_r['classes'][max_i]
            scene_class = test_data['Y'][idx]

            if pred_class == scene_class:
                corr += 1.0

        acc.append((alpha, corr/length))
    
    acc = sorted(acc, key=lambda t: t[0])
    x, y = map(list,zip(*acc))

    return x, y

if __name__ == '__main__':
    declarative = Declarative()

    train_data, test_data, length = open_indoor_test(declarative)
    
    svm = SVC(kernel='rbf', probability=True).fit(train_data['X'], train_data['Y'])

    x, y = test_indoor(svm, declarative, train_data, test_data, [0.0])

    print(y)

