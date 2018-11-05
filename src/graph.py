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
    def __init__(self, arch = 'resnet-18-ImageNet'):

        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.ann_path = os.path.join(self.config['path'], 'data', 'sun2012_ann.json')
        if os.path.exists(self.ann_path):
            with open(self.ann_path, 'r') as fp:
                self.annotations = json.load(fp)
        else:
            print('Annotations data not found!')

        self.graph_path = os.path.join(self.config['path'], 'data', 'sun2012_ann.json')
        if os.path.exists(self.graph_path):
            print('Loading graph')
            with open(self.graph_path, 'r') as fp:
                self.graph = json.load(fp)
        else:
            print('Graph data not found!')
            self.graph = {}

            try:
                from lib.img_to_vec import Img2Vec
            except ImportError:
                from src.img_to_vec import Img2Vec

            self.img2vec = Img2Vec(model = arch)

            self.train_sun()

    def img_channels(self, img):
        # Reshape for 3-channels
        if len(np.array(img).shape) != 3:
            img = np.stack((img,)*3, axis=-1)
            img = Image.fromarray(np.uint8(img))
        elif np.array(img).shape[2] > 3:
            img = img.convert('RGB')
            img = np.asarray(img, dtype=np.float32)
            img = img[:, :, :3]
            img = Image.fromarray(np.uint8(img))

        return img

    def train_sun(self):
        # Create graph of SUN 2012
        print('Training graph - SUN 2012')

        pattern = '*.jpg'
        root = os.path.join(self.config['path'], 'data', 'SUN2012pascalformat', 'JPEGImages')

        for path, subdirs, files in os.walk(root):
            length = len(files)

            for idx, name in enumerate(files):
                sys.stdout.write(name + ' -- ' + str(idx+1) + '/' + str(length) + '\r')

                if fnmatch(name, pattern):
                    img = Image.open(os.path.join(path, name))
                    img_name = os.path.splitext(name)[0] # Remove extension
                    width, height = img.size

                    img = self.img_channels(img)

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

                            box = (x1, y1, x2, y2)
                            region = img.crop(box)
                            vec = self.img2vec.get_vec(region)

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

                            box = (x1, y1, x2, y2)
                            region = img.crop(box)
                            vec = self.img2vec.get_vec(region)

                            node = self.graph.get(objects['name'])
                            if node:
                                node['vec'] = (np.array(node['vec']) + vec).tolist()
                            else:
                                self.graph[objects['name']] = {}
                                self.graph[objects['name']]['vec'] = vec.tolist()

                            break

        with open(os.path.join(self.config['path'], 'data', 'graph.json'), 'w') as fp:
            json.dump(self.graph, fp, sort_keys=True, indent=4)

    def relation_obj(self, cx_a, cy_a, img_name, idx, height):
        # Calculate angle between objects and identify their neighbours
        angles = {}
        neighbours = {}

        objects = self.annotations[img_name]['annotation']['object']
        for i, obj in enumerate(objects):
            if i == idx:
                continue

            try:
                if self.config['dataset'] == 'indoor':
                    x = []
                    y = []
                    for p in obj['polygon']['pt']:
                        x.append(int(p['x']))
                        y.append(int(p['y']))

                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)
                elif self.config['dataset'] == 'sun2012':
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
