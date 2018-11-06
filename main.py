## Henrique X. Goulart

import json
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import spatial
from sklearn.svm import SVC

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

def test_scene_indoor(declarative):
        # Test on MIT Indoor 67
        print('Testing Declarative Memory - Scenes Vectors - MIT Indoor 67')

        acc1 = 0.0

        path_test = os.path.join(declarative.config['path'], 'data', 'TestImages.txt')
        length = len(open(path_test).readlines())

        test_indoor_path = os.path.join(declarative.config['path'], 'data', 'test_indoor.pkl')
        test_data = None
        if os.path.exists(test_indoor_path):
            with open(test_indoor_path, 'rb') as fp:
                test_data = pickle.load(fp)

        train_indoor_path = os.path.join(declarative.config['path'], 'data', 'train_indoor.pkl')
        train_data = None
        if os.path.exists(train_indoor_path):
            with open(train_indoor_path, 'rb') as fp:
                train_data = pickle.load(fp)

        svm = SVC(kernel='rbf').fit(train_data['X'],train_data['Y'])

        X_test = []
        Y_test = []

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                sys.stdout.write('Testing... ' + str(idx+1) + '/' + str(length) + '\r')

                if not test_data:
                    scene_class = Path(line).parts[0].strip() # Get name supervision from path

                    path = os.path.join(declarative.config['path'], 'data', 'MITImages', line.strip())
                    img = Image.open(path)
                    img = declarative.img_channels(img)

                    vec = declarative.img2vec.get_vec(img)

                    X_test.append(vec)
                    Y_test.append(scene_class)
                else:
                    vec = test_data['X'][idx]
                    scene_class = test_data['Y'][idx]

                distance = np.inf
                pred_class = ''

                for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                    d = spatial.distance.cosine(vec, class_vec)

                    if d < distance:
                        pred_class = class_name
                        distance = d

                if pred_class == scene_class:
                    acc1 += 1.0

                # region_vectors = declarative.extract_regions(img)
                #
                # objects = []
                # for reg in region_vectors:
                #     objects.append(declarative.graph.get_obj(reg))
                # objects = [*set(objects)]
                #
                # neighbours = []
                # for obj in objects:
                #     neighbours += declarative.graph.get_subgraph(obj)
                # neighbours = [*set(neighbours)]
                #
                # for co_occurrence in declarative.declarative_data['co_occurrences']:
                #     if all(elem in co_occurrence['co_occurrence'] for elem in neighbours):
                #         distance = np.inf
                #         pred_class = ''
                #
                #         for class_name, class_vec in declarative.declarative_data['scene_vectors'].items():
                #             d = spatial.distance.cosine(co_occurrence['scene_vec'], class_vec)
                #
                #             if d < distance:
                #                 pred_class = class_name
                #                 distance = d
                #
                #         if pred_class == scene_class:
                #             acc2 += 1.0
                #
                #         break


        if not os.path.exists(test_indoor_path):
            test_data = {'X':X_test, 'Y':Y_test}
            with open(test_indoor_path, 'wb') as fp:
                pickle.dump(test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

        acc2 = svm.score(test_data['X'], test_data['Y'])

        print('Accuracy on MIT Indoor 67: {} / {}'.format(acc1/length, acc2))

if __name__ == '__main__':
    graph = Graph(arch = config['arch_obj'])
    declarative = Declarative(graph)

    test_scene_indoor(declarative)
