# Spatial Memory
## Henrique X. Goulart

import os
import sys
import json
import pickle
import numpy as np
import multiprocessing as mp
from PIL import Image
from pathlib import Path

class Spatial():
    def __init__(self, train_data, test_data):
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        try:
            from lib.img_to_vec import Img2Vec
        except ImportError:
            from src.img_to_vec import Img2Vec
        self.img2vec = Img2Vec(model = self.config['arch_obj'])

        self.train_indoor_path_spatial = os.path.join(self.config['path'], 'data', 'train_indoor_spatial.pkl')
        if os.path.exists(self.train_indoor_path_spatial):
            print('Loading Spatial (train)...')
            with open(self.train_indoor_path_spatial, 'rb') as fp:
                self.train_data = pickle.load(fp)
        else:
            print('Spatial train data not found!')
            self.train_indoor(train_data)

        self.test_indoor_path_spatial = os.path.join(self.config['path'], 'data', 'test_indoor_spatial.pkl')
        if os.path.exists(self.test_indoor_path_spatial):
            print('Loading Spatial (test)...')
            with open(self.test_indoor_path_spatial, 'rb') as fp:
                self.test_data = pickle.load(fp)
        else:
            print('Spatial test data not found!')
            self.test_indoor(test_data)

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

        # 2x2
        box = (0, 0, int(width/2), int(height/2)) # 1st region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/2), 0, int(width), int(height/2)) # 2nd region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (0, int(height/2), int(width/2), int(height)) # 3rd region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/2), int(height/2), int(width), int(height)) # 4th region
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        # 3x3
        box = (0, 0, int(width/3), int(height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/3), 0, int(2*width/3), int(height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(2*width/3), 0, int(width), int(height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (0, int(height/3), int(width/3), int(2*height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/3), int(height/3), int(2*width/3), int(2*height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(2*width/3), int(height/3), int(width), int(2*height/3))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (0, int(2*height/3), int(width/3), int(height))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(width/3), int(2*height/3), int(2*width/3), int(height))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        box = (int(2*width/3), int(2*height/3), int(width), int(height))
        processes_ext.append(mp.Process(target=self.region_vec, args=(box, img, output_ext)))

        for p in processes_ext:
            p.start()
        for p in processes_ext:
            p.join()

        vec = [output_ext.get() for p in processes_ext]
        vec = [float(sum(l))/len(l) for l in zip(*vec)]

        return vec

    def train_indoor(self, td):
        # Train Spatial Memory on MIT Indoor 67
        print('Training Spatial Memory - MIT Indoor 67')

        self.train_data = td.copy()

        path_train = os.path.join(self.config['path'], 'data', 'TrainImages.txt')

        length = len(open(path_train).readlines())

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get class supervision from path

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    img = Image.open(path)
                    img = self.img_channels(img)

                    vec = self.extract_regions(img)

                    self.train_data['X'][idx] = list(vec) + list(self.train_data['X'][idx])
                except:
                    print('Error at {}'.format(line))
                    return
        with open(self.train_indoor_path_spatial, 'wb') as fp:
            pickle.dump(self.train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_indoor(self, td):
        # Test Spatial Memory on MIT Indoor 67
        print('Testing Spatial Memory - MIT Indoor 67')

        self.test_data = td.copy()

        path_test = os.path.join(self.config['path'], 'data', 'TestImages.txt')

        length = len(open(path_test).readlines())

        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    scene_class = Path(line).parts[0].strip() # Get class supervision from path

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    img = Image.open(path)
                    img = self.img_channels(img)

                    vec = self.extract_regions(img)

                    self.test_data['X'][idx] = list(vec) + list(self.test_data['X'][idx])
                except:
                    print('Error at {}'.format(line))
                    return
        with open(self.test_indoor_path_spatial, 'wb') as fp:
            pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)