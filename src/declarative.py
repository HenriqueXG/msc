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
    def __init__(self):
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        try:
            from lib.img_to_vec import Img2Vec
        except ImportError:
            from src.img_to_vec import Img2Vec
        self.img2vec = Img2Vec(model = self.config['arch_scene'])

        self.declarative_path = os.path.join(self.config['path'], 'data', 'declarative_data.json')
        self.train_indoor_path = os.path.join(self.config['path'], 'data', 'train_indoor_declarative.pkl')
        if (os.path.exists(self.declarative_path) or os.path.exists(self.train_indoor_path)) and self.config['dataset'] == 'indoor':
            print('Loading declarative data (train)...')
            with open(self.declarative_path, 'r') as fp:
                self.declarative_data = json.load(fp)
            with open(self.train_indoor_path, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            self.train_scene_indoor()

        self.test_indoor_path = os.path.join(self.config['path'], 'data', 'test_indoor_declarative.pkl')
        if os.path.exists(self.test_indoor_path) and self.config['dataset'] == 'indoor':
            print('Loading declarative data (test)...')
            with open(self.test_indoor_path, 'rb') as fp:
                self.test_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            self.test_scene_indoor()

        if os.path.exists(self.declarative_path) and self.config['dataset'] == 'sun397':
            print('Loading declarative data...')
            with open(self.declarative_path, 'r') as fp:
                self.declarative_data = json.load(fp)
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

    def train_scene_sun397(self):
        # Train SUN397 scene vectors
        print('Training Declarative Memory - Scenes - SUN 397')

        root = os.path.join(self.config['path'], 'data', 'SUN397')

        for path, _, files in os.walk(root):
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

        with open(self.declarative_path, 'w') as fp:
            json.dump(self.declarative_data, fp, sort_keys=True, indent=4)

        self.train_data = {'X':X_train, 'Y':Y_train}
        with open(self.train_indoor_path, 'wb') as fp:
            pickle.dump(self.train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_scene_indoor(self):
        # Test MIT Indoor 67 scene vectors
        print('Testing Declarative Memory - Scenes - MIT Indoor 67')

        path_test = os.path.join(self.config['path'], 'data', 'TestImages.txt')
        length = len(open(path_test).readlines())

        X_test = []
        Y_test = []

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get name supervision from path

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    img = Image.open(path)
                    img = self.img_channels(img)

                    vec = self.img2vec.get_vec(img)

                    X_test.append(vec)
                    Y_test.append(scene_class)
                except:
                    print('Error at {}'.format(line))
                    break

        self.test_data = {'X':X_test, 'Y':Y_test}
        with open(self.test_indoor_path, 'wb') as fp:
            pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
