# Graph for PAM and Spatial Memory
## Henrique X. Goulart

import json
import sys
import os
import gluoncv
import numpy as np
from PIL import Image
from fnmatch import fnmatch
from pathlib import Path
from gluoncv import model_zoo, data, utils

class Graph():
    def __init__(self):
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.net = model_zoo.get_model(self.config['arch_pam'], pretrained=True) # pre-trained neural network with COCO dataset
        self.coco_classes = self.net.classes # Vector of COCO classes

        self.graph_path = os.path.join(self.config['path'], 'data', 'graph_indoor.json')
        if os.path.exists(self.graph_path):
            print('Loading graph...')
            with open(self.graph_path, 'r') as fp:
                self.graph = json.load(fp)
        else:
            print('Graph data not found!')
            self.graph = {}
            self.train_indoor()

    def train_indoor(self):
        # Create graph of MIT Indoor 67
        print('Training graph - MIT Indoor 67')

        path_train = os.path.join(self.config['path'], 'data', 'TrainImages.txt')
        length = len(open(path_train).readlines())

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get class supervision from path

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    im_fname = utils.download('', path = path)
                    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

                    box_ids, sscores, bboxes = self.net(x)

                    boxes = []
                    scores = []
                    ids = []
                    for i in range(len(sscores[0])):
                        if sscores[0][i][0] > float(self.config['pam_threshold']): # Choose only detections that have confident higher than a threshold
                            boxes.append(bboxes.asnumpy()[0][i])
                            scores.append(float(sscores.asnumpy()[0][i][0]))
                            ids.append(int(box_ids.asnumpy()[0][i][0]))
                    
                    ids = [*set(ids)]
                    
                    node = self.graph.get(scene_class)
                    if not node:
                        self.graph[scene_class] = {}
                        node = self.graph.get(scene_class)

                    for i in ids:
                        try:
                            node[i] += 1
                        except KeyError:
                            node[i] = 1
                except:
                    print('Error at {}'.format(line))
                    return

        with open(self.graph_path, 'w') as fp:
            json.dump(self.graph, fp, sort_keys=True, indent=4)