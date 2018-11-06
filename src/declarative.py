# Declarative Memory
## Henrique X. Goulart

import json
import os
import sys
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from fnmatch import fnmatch

class Declarative():
    def __init__(self, graph):

        self.graph = graph

        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.ann_path = os.path.join(self.config['path'], 'data', 'sun2012_ann.json')
        if os.path.exists(self.ann_path):
            with open(self.ann_path, 'r') as fp:
                self.annotations = json.load(fp)

        try:
            from lib.img_to_vec import Img2Vec
        except ImportError:
            from src.img_to_vec import Img2Vec

        self.img2vec = Img2Vec(model = self.config['arch_scene'])
        self.reg2vec = Img2Vec(model = self.config['arch_obj'])

        self.declarative_path = os.path.join(self.config['path'], 'data', 'declarative_data.json')
        if os.path.exists(self.declarative_path):
            print('Loading declarative data')
            with open(self.declarative_path, 'r') as fp:
                self.declarative_data = json.load(fp)

            if not self.declarative_data.get('co_occurrences'):
                self.train_obj()
            if not self.declarative_data.get('scene_vectors'):
                self.declarative_data['scene_vectors'] = {}
                if self.config['dataset'] == 'indoor':
                    self.train_scene_indoor()
                elif self.config['dataset'] == 'sun397':
                    self.train_scene_sun397()
        else:
            print('Declarative data not found!')
            self.declarative_data = {}

            self.train_obj()

            self.declarative_data['scene_vectors'] = {}
            if self.config['dataset'] == 'indoor':
                self.train_scene_indoor()
            elif self.config['dataset'] == 'sun397':
                self.train_scene_sun397()

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

    def find_img_sun(self, pattern):
        root = os.path.join(self.config['path'], 'data', 'SUN2012pascalformat', 'JPEGImages')

        img = None
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, pattern):
                    img = Image.open(os.path.join(path, name))
                    return img

    def train_obj(self):
        # Train object co-occurences
        print('Training Declarative Memory - Objects')

        co_occurrences = []

        length = len(self.annotations.items())

        for idx, (k, v) in enumerate(self.annotations.items()):
            sys.stdout.write(k + ' -- ' + str(idx+1) + '/' + str(length) + '\r')

            objects = v['annotation']['object']

            co_occurrence = []

            for obj in objects:
                try:
                    co_occurrence.append(obj['name'])
                except TypeError:
                    co_occurrence.append(objects['name'])
                    break

            co_occurrence = {'co_occurrence':[*set(co_occurrence)]}

            pattern = v['annotation']['filename']
            img = self.find_img_sun(pattern)
            img = self.img_channels(img)

            vec = self.img2vec.get_vec(img)
            co_occurrence['scene_vec'] = vec.tolist()

            co_occurrences.append(co_occurrence)

        self.declarative_data['co_occurrences'] = co_occurrences
        with open(os.path.join(self.config['path'], 'data', 'declarative_data.json'), 'w') as fp:
            json.dump(self.declarative_data, fp, sort_keys=True, indent=4)

    def train_scene_sun397(self):
        # Train SUN397 scene vectors
        print('Training Declarative Memory - Scenes - SUN 397')

        root = os.path.join(self.config['path'], 'data', 'SUN397')

        for path, subdirs, files in os.walk(root):
            length = len(files)

            for idx, name in enumerate(files):
                if name.endswith(('.jpg', '.jpeg', '.gif', '.png')):
                    try:
                        sys.stdout.write(name + ' -- ' + str(idx+1) + '/' + str(length) + '\r')

                        rel_parts = Path(path).relative_to(root).parts[1:] # Get name supervision from path
                        scene_class = Path(*list(rel_parts)).as_posix() # Scene class

                        img = Image.open(os.path.join(path, name))
                        img = self.img_channels(img)

                        vec = self.img2vec.get_vec(img)

                        scene_vec = self.declarative_data['scene_vectors'].get(scene_class)
                        if scene_vec:
                            self.declarative_data['scene_vectors'][scene_class] = (np.array(scene_vec) + vec).tolist()
                        else:
                            self.declarative_data['scene_vectors'][scene_class] = vec.tolist()
                    except:
                        print('Error at {}'.format(name))

        with open(os.path.join(self.config['path'], 'data', 'declarative_data.json'), 'w') as fp:
            json.dump(self.declarative_data, fp, sort_keys=True, indent=4)

    def train_scene_indoor(self):
        # Train MIT Indoor 67 scene vectors
        print('Training Declarative Memory - Scenes - MIT Indoor 67')

        path_train = os.path.join(self.config['path'], 'data', 'TrainImages.txt')
        length = len(open(path_train).readlines())

        X_train = []
        Y_train = []

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get name supervision from path

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    img = Image.open(path)
                    img = self.img_channels(img)

                    vec = self.img2vec.get_vec(img)

                    X_train.append(vec)
                    Y_train.append(scene_class)

                    scene_vec = self.declarative_data['scene_vectors'].get(scene_class)
                    if scene_vec:
                        self.declarative_data['scene_vectors'][scene_class] = (np.array(scene_vec) + vec).tolist()
                    else:
                        self.declarative_data['scene_vectors'][scene_class] = vec.tolist()
                except:
                    print('Error at {}'.format(line))
                    break

        with open(os.path.join(self.config['path'], 'data', 'declarative_data.json'), 'w') as fp:
            json.dump(self.declarative_data, fp, sort_keys=True, indent=4)

        train_data = {'X':X_train, 'Y':Y_train}
        with open(os.path.join(self.config['path'], 'data', 'train_indoor.pkl'), 'wb') as fp:
            pickle.dump(train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_regions(self, img):
        # Extract Sub-images vectors
        region_vectors = []
        width, height = img.size

        box = (0, 0, int(width/2), int(height/2)) # 1st region
        region = img.crop(box)
        vec = self.reg2vec.get_vec(region)
        region_vectors.append(vec)

        box = (int(width/2), 0, int(width), int(height/2)) # 2nd region
        region = img.crop(box)
        vec = self.reg2vec.get_vec(region)
        region_vectors.append(vec)

        box = (0, int(height/2), int(width/2), int(height)) # 3rd region
        region = img.crop(box)
        vec = self.reg2vec.get_vec(region)
        region_vectors.append(vec)

        box = (int(width/2), int(height/2), int(width), int(height)) # 4th region
        region = img.crop(box)
        vec = self.reg2vec.get_vec(region)
        region_vectors.append(vec)

        return region_vectors

