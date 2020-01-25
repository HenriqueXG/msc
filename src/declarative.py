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

        # MIT Indoor
        self.train_indoor_path_declarative = os.path.join(self.config['path'], 'data', 'train_indoor_declarative.pkl')
        if os.path.exists(self.train_indoor_path_declarative) and self.config['dataset'] == 'indoor':
            print('Loading declarative data (train)...')
            with open(self.train_indoor_path_declarative, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            self.train_indoor()

        self.test_indoor_path_declarative = os.path.join(self.config['path'], 'data', 'test_indoor_declarative.pkl')
        if os.path.exists(self.test_indoor_path_declarative) and self.config['dataset'] == 'indoor':
            print('Loading declarative data (test)...')
            with open(self.test_indoor_path_declarative, 'rb') as fp:
                self.test_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            self.test_indoor()

        # SUN 397
        self.train_sun397_path_declarative = os.path.join(self.config['path'], 'data', 'train_sun397_declarative_{:0>2d}.pkl'.format(self.config['sun397_it']))
        if os.path.exists(self.train_sun397_path_declarative) and self.config['dataset'] == 'sun397':
            print('Loading declarative data (train)...')
            with open(self.train_sun397_path_declarative, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'sun397':
            self.train_sun397()

        self.test_sun397_path_declarative = os.path.join(self.config['path'], 'data', 'test_sun397_declarative_{:0>2d}.pkl'.format(self.config['sun397_it']))
        if os.path.exists(self.test_sun397_path_declarative) and self.config['dataset'] == 'sun397':
            print('Loading declarative data (test)...')
            with open(self.test_sun397_path_declarative, 'rb') as fp:
                self.test_data = pickle.load(fp)
        elif self.config['dataset'] == 'sun397':
            self.test_sun397()

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

    def train_sun397(self):
        # Train SUN397 scene vectors
        print('Training Declarative Memory - SUN 397')

        print('Training_{:0>2d}.txt'.format(self.config['sun397_it']))

        path_train = os.path.join(self.config['path'], 'data', 'SUNPartitions', 'Training_{:0>2d}.txt'.format(self.config['sun397_it']))
        root = os.path.join(self.config['path'], 'data', 'SUN397')
        length = len(open(path_train).readlines())

        X_train = []
        Y_train = []

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = line.split('/')[-2] # Get name supervision from path

                    path = root + line.strip()

                    img = Image.open(path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = self.img_channels(img)

                    vec = self.img2vec.get_vec(img)

                    X_train.append(vec)
                    Y_train.append(scene_class)
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return
        print('')

        self.train_data = {'X':X_train, 'Y':Y_train}
        with open(self.train_sun397_path_declarative, 'wb') as fp:
            pickle.dump(self.train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_sun397(self):
        # Test SUN397 scene vectors
        print('Testing Declarative Memory - SUN 397')

        print('Testing_{:0>2d}.txt'.format(self.config['sun397_it']))

        path_test = os.path.join(self.config['path'], 'data', 'SUNPartitions', 'Testing_{:0>2d}.txt'.format(self.config['sun397_it']))
        root = os.path.join(self.config['path'], 'data', 'SUN397')
        length = len(open(path_test).readlines())

        X_test = []
        Y_test = []

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = line.split('/')[-2] # Get name supervision from path

                    path = root + line.strip()

                    img = Image.open(path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = self.img_channels(img)

                    vec = self.img2vec.get_vec(img)

                    X_test.append(vec)
                    Y_test.append(scene_class)
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return
        print('')

        self.test_data = {'X':X_test, 'Y':Y_test}
        with open(self.test_sun397_path_declarative, 'wb') as fp:
            pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def train_indoor(self):
        # Train Declarative Memory on MIT Indoor 67 scene vectors
        print('Training Declarative Memory - MIT Indoor 67')

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
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return

        self.train_data = {'X':X_train, 'Y':Y_train}
        with open(self.train_indoor_path_declarative, 'wb') as fp:
            pickle.dump(self.train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_indoor(self):
        # Test Declarative Memory on MIT Indoor 67 scene vectors
        print('Testing Declarative Memory - MIT Indoor 67')

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
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return

        self.test_data = {'X':X_test, 'Y':Y_test}
        with open(self.test_indoor_path_declarative, 'wb') as fp:
            pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
