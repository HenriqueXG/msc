# Graph for PAM and Spatial Memory
## Henrique X. Goulart

import json
import sys
import os
import sys
import math
import numpy as np
import multiprocessing as mp
from PIL import Image
from scipy import spatial
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

        self.graph_path = os.path.join(self.config['path'], 'data', 'sun2012_graph.json')
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

        with open(os.path.join(self.config['path'], 'data', 'sun2012_graph.json'), 'w') as fp:
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

    def get_obj(self, vec, output):
        # Get the nearest vector on graph
        distance = np.inf
        obj = ''

        for k, v in self.graph.items():
            obj_vec = v['vec']

            d = spatial.distance.cosine(vec, obj_vec)

            if d < distance:
                distance = d
                obj = k

        output.put(obj)

    def get_subgraph(self, obj):
        # Extract node neighbours
        neighbours = []

        for ngb, freq in self.graph[obj]['neighbours'].items():
            neighbours.append(ngb)

        return neighbours
    
    def region_vec(self, box, img, output):
        # Region to vec
        region = img.crop(box)
        vec = self.img2vec.get_vec(region)

        output.put(vec)

    def extract_regions(self, img):
        # Extract Sub-images vectors
        width, height = img.size

        processes_ext = []
        output_ext = mp.Queue()

        box = (0, 0, int(width/2), int(height/2)) # 1st region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/2), 0, int(width), int(height/2)) # 2nd region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (0, int(height/2), int(width/2), int(height)) # 3rd region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/2), int(height/2), int(width), int(height)) # 4th region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        for p in processes_ext:
            p.start()
        for p in processes_ext:
            p.join()

        region_vectors = [output_ext.get() for p in processes_ext]
        return region_vectors
