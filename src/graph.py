# Graph for PAM and Spatial Memory
## Henrique X. Goulart

import json
import sys
import os
import sys
import math
import numpy as np
from PIL import Image
from fnmatch import fnmatch

class Graph():
    def __init__(self):

        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.ann_path = os.path.join(self.config['path'], 'data/sun2012_ann.json')
        if os.path.exists(self.ann_path):
            with open(self.ann_path, 'r') as fp:
                self.annotations = json.load(fp)
        else:
            print('Annotations data not found!')

        self.graph_path = os.path.join(self.config['path'], 'data/graph.json')
        if os.path.exists(self.graph_path):
            print('Loading Graph')
            with open(self.graph_path, 'r') as fp:
                self.graph = json.load(fp)
        else:
            print('Graph data not found!')
            self.graph = {}

            sys.path.append(os.path.join(self.config['path'], 'src'))
            from img_to_vec import Img2Vec
            arch = 'resnet-18'
            self.img2vec = Img2Vec(model = arch)

            self.train()

    def train(self):
        # Create Graph
        print('Training Graph')

        pattern = '*.jpg'
        root = os.path.join(self.config['path'], 'data/SUN2012pascalformat/JPEGImages/')

        for path, subdirs, files in os.walk(root):
            for idx, name in enumerate(files):
                sys.stdout.write(name + ' -- ' + str(idx+1) + '/' + str(len(files)) + '\r')

                if fnmatch(name, pattern):
                    img = Image.open(os.path.join(path, name))
                    img_name = name.replace('.jpg','')
                    width, height = img.size
                    if np.array(img).shape[2] != 3:
                        img = np.resize(img, (height, width, 3))

                    objects = self.annotations[img_name]['annotation']['object']
                    for i, obj in enumerate(objects):
                        stop = False
                        try:
                            x1 = int(obj['bndbox']['xmin'])
                            x2 = int(obj['bndbox']['xmax'])
                            y1 = int(obj['bndbox']['ymin'])
                            y2 = int(obj['bndbox']['ymax'])

                            dx = x2 - x1
                            dy = y2 - y1
                            cx = int(dx/2) + x1 # center of mass x
                            cy = height - (int(dy/2) + y1) # center of mass y

                            angles, neighbours = self.relation_obj(cx, cy, img_name, i, height)

                            box = (x1,y1,x2,y2)
                            region = img.crop(box)
                            vec = self.img2vec.get_vec(region)
                            # vec = np.array([0,0])

                            node = self.graph.get(obj['name'])
                            if node:
                                node['vec'] = (np.array(node['vec']) + vec).tolist()

                                for k1, v1 in node['angles'].items():
                                    for k2, v2 in angles.items():
                                        if k1 == k2:
                                            node['angles'][k1] += angles[k2]
                                            node['angles'][k1] = [*set(node['angles'][k1])]
                                for k, v in angles.items():
                                    if k not in node['angles']:
                                        node['angles'][k] = v

                                for k1, v1 in node['neighbours'].items():
                                    for k2, v2 in neighbours.items():
                                        if k1 == k2:
                                            node['neighbours'][k1] += neighbours[k2]
                                for k, v in neighbours.items():
                                    if k not in node['neighbours']:
                                        node['neighbours'][k] = v

                            else:
                                self.graph[obj['name']] = {}
                                self.graph[obj['name']]['vec'] = vec.tolist()
                                self.graph[obj['name']]['angles'] = angles
                                self.graph[obj['name']]['neighbours'] = neighbours

                        except TypeError:
                            x1 = int(objects['bndbox']['xmin'])
                            x2 = int(objects['bndbox']['xmax'])
                            y1 = int(objects['bndbox']['ymin'])
                            y2 = int(objects['bndbox']['ymax'])

                            box = (x1,y1,x2,y2)
                            region = img.crop(box)
                            vec = self.img2vec.get_vec(region)

                            node = self.graph.get(objects['name'])
                            if node:
                                node['vec'] = (np.array(node['vec']) + vec).tolist()
                            else:
                                self.graph[objects['name']] = {}
                                self.graph[objects['name']]['vec'] = vec.tolist()

                            break



        with open(os.path.join(self.config['path'], 'data/graph.json'), 'w') as fp:
            json.dump(self.graph, fp, sort_keys=True, indent=4)


    def relation_obj(self, cx_a, cy_a, img_name, idx, height):
        # Calculate angle between objects identify their neighbours
        angles = {}
        neighbours = {}

        objects = self.annotations[img_name]['annotation']['object']
        for i, obj in enumerate(objects):
            if i == idx:
                continue

            try:
                x1 = int(obj['bndbox']['xmin'])
                x2 = int(obj['bndbox']['xmax'])
                y1 = int(obj['bndbox']['ymin'])
                y2 = int(obj['bndbox']['ymax'])
            except TypeError:
                return {},{}

            dx = x2 - x1
            dy = y2 - y1
            cx_b = int(dx/2) + x1 # center of mass x
            cy_b = height - (int(dy/2) + y1) # center of mass y

            dcx = cx_b - cx_a
            dcy = cy_b - cy_a

            angle = math.degrees(math.atan2(dcy,dcx) % (2 * np.pi)) # angle 0-360 degrees

            try:
                angles[obj['name']].append(angle)
            except KeyError:
                angles[obj['name']] = [angle]

            try:
                neighbours[obj['name']] += 1
            except KeyError:
                neighbours[obj['name']] = 1

        return angles, neighbours
