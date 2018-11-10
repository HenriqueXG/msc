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

def test_indoor_graph(declarative, graph, test_data, length, output):
    # Graph test on MIT Indoor 67

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')

    corr = 0.0

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, line in enumerate(archive):
            sys.stdout.write('Testing graph... ' + str(idx+1) + '/' + str(length) + '\r')
            
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
    print('Testing cosine...')

    path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')

    corr = 0.0
    pred = {}
    pred['model'] = 'cosine'

    classes_ = [class_name for class_name, _ in declarative.declarative_data['scene_vectors'].items()]
    pred['classes'] = classes_

    with open(path_test, 'r', encoding='ISO-8859-1') as archive:
        for idx, _ in enumerate(archive):
            vec = test_data['X'][idx]
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

def test_indoor(svm, declarative, graph, train_data, test_data, alphas, output_test):
    # Testing - Scenes Vectors - MIT Indoor 67
    for alpha in alphas:
        output = mp.Queue()

        processes = []
        processes.append(mp.Process(target=test_indoor_cosine, args=(declarative, test_data, length, output)))
        processes.append(mp.Process(target=test_indoor_svm, args=(svm, train_data, test_data, output)))
        # processes.append(mp.Process(target=test_indoor_graph, args=(declarative, graph, test_data, length, output)))

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

        output_test.put((alpha, corr/length))

if __name__ == '__main__':
    # graph = Graph(arch = config['arch_obj'])
    declarative = Declarative()

    print('Testing - Scenes Vectors - MIT Indoor 67')
    train_data, test_data, length = open_indoor_test(declarative)
    
    svm = SVC(kernel='rbf', probability=True).fit(train_data['X'], train_data['Y'])

    output_test = mp.Queue()
    NB_JOBS = mp.cpu_count()
    xs = np.arange(0.0, 1.01, 0.01)
    alpha_parts = np.array_split(xs, NB_JOBS)

    processes = []
    for part in alpha_parts:
        processes.append(mp.Process(target=test_indoor, args=(svm, declarative, None, train_data, test_data, part, output_test)))

    for p in processes:
        p.start()

    acc = [output_test.get() for p in processes]

    for p in processes:
        p.join()


    acc = sorted(acc, key=lambda t: t[0])
    x, y = map(list,zip(*acc))

    plt.plot(x, y)
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.show()

