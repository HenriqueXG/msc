# PAM/EPAM
## Henrique X. Goulart

import json
import sys
import os
import gluoncv
import pickle
import numpy as np
from PIL import Image
from fnmatch import fnmatch
from pathlib import Path
from gluoncv import model_zoo, data, utils

class PAM():
    def __init__(self):
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.net = model_zoo.get_model(self.config['arch_pam'], pretrained=True) # Pre-trained neural network with COCO dataset
        self.coco_classes = self.net.classes # Vector of COCO classes

        try:
            from lib.img_to_vec import Img2Vec
        except ImportError:
            from src.img_to_vec import Img2Vec
        self.img2vec = Img2Vec(model = self.config['arch_obj']) # Pre-trained neural network with ImageNet dataset

        # MIT Indoor
        self.train_indoor_path_pam = os.path.join(self.config['path'], 'data', 'train_indoor_pam.pkl')
        if os.path.exists(self.train_indoor_path_pam) and self.config['dataset'] == 'indoor':
            print('Loading PAM (train)...')
            with open(self.train_indoor_path_pam, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            print('PAM train data not found!')
            self.train_indoor()

        self.test_indoor_path_pam = os.path.join(self.config['path'], 'data', 'test_indoor_pam.pkl')
        if os.path.exists(self.test_indoor_path_pam) and self.config['dataset'] == 'indoor':
            print('Loading PAM (test)...')
            with open(self.test_indoor_path_pam, 'rb') as fp:
                self.test_data = pickle.load(fp)
        elif self.config['dataset'] == 'indoor':
            print('PAM test data not found!')
            self.test_indoor()

        # SUN 397
        self.train_sun397_path_pam = os.path.join(self.config['path'], 'data', 'train_sun397_pam.pkl')
        if os.path.exists(self.train_sun397_path_pam) and self.config['dataset'] == 'sun397':
            print('Loading PAM data (train)...')
            with open(self.train_sun397_path_pam, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'sun397':
            print('PAM train data not found!')
            self.train_sun397()

        self.test_sun397_path_pam = os.path.join(self.config['path'], 'data', 'test_sun397_pam.pkl')
        if os.path.exists(self.test_sun397_path_pam) and self.config['dataset'] == 'sun397':
            print('Loading PAM data (test)...')
            with open(self.test_sun397_path_pam, 'rb') as fp:
                self.test_data = pickle.load(fp)
        elif self.config['dataset'] == 'sun397':
            print('PAM test data not found!')
            self.test_sun397()

        # SUN 397
        self.train_sun397_path_pam = os.path.join(self.config['path'], 'data', 'train_sun397_pam.pkl')
        if os.path.exists(self.train_sun397_path_pam) and self.config['dataset'] == 'sun397':
            print('Loading declarative data (train)...')
            with open(self.train_sun397_path_pam, 'rb') as fp:
                self.train_data = pickle.load(fp)
        elif self.config['dataset'] == 'sun397':
            self.train_sun397()

        self.test_sun397_path_pam = os.path.join(self.config['path'], 'data', 'test_sun397_pam.pkl')
        if os.path.exists(self.test_sun397_path_pam) and self.config['dataset'] == 'sun397':
            print('Loading declarative data (test)...')
            with open(self.test_sun397_path_pam, 'rb') as fp:
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
        print('Training PAM - SUN 397')

        for i in range(10):
            print('Training_{:0>2d}.txt'.format(i+1))

            path_train = os.path.join(self.config['path'], 'data', 'SUNPartitions', 'Training_{:0>2d}.txt'.format(i+1))
            root = os.path.join(self.config['path'], 'data', 'SUN397')
            length = len(open(path_train).readlines())

            X_train = []

            with open(path_train, 'r', encoding='ISO-8859-1') as archive:
                for idx, line in enumerate(archive):
                    try:
                        sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                        path = root + line.strip()

                        ## YOLO

                        im_fname = utils.download('', path = path)
                        img = Image.open(im_fname)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(im_fname)
                        x, img = data.transforms.presets.rcnn.load_test(im_fname)

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
                        
                        ids_bool = list([0] * len(self.coco_classes))

                        for i in ids:
                            ids_bool[i] = 1

                        ## ImageNet

                        img = self.img_channels(img)
                        img = Image.fromarray(np.uint8(img))

                        vec = []
                        for b in boxes:
                            x1 = b[0]
                            y1 = b[1]
                            x2 = b[2]
                            y2 = b[3]

                            box = (x1, y1, x2, y2)
                            region = img.crop(box)
                            vec.append(self.img2vec.get_vec(region).tolist())
                        vec = [float(sum(l))/len(l) for l in zip(*vec)]

                        if len(vec) == 0:
                            vec = np.zeros(self.img2vec.layer_output_size_pool).tolist()

                        ## Join vectors

                        join_vec = ids_bool + list(vec)

                        X_train.append(join_vec)
                    except Exception as e:
                        with open(os.path.join(self.config['path'], 'error_training_{:0>2d}.log'.format(i+1)), 'a') as log_file:
                            log_file.write('Error at {} \n'.format(line))
                            log_file.write(str(e) + '\n')
                            log_file.write('@@@\n')
                        pass
            print('')

            with open(self.train_sun397_path_pam + '_training_{:0>2d}'.format(i+1), 'wb') as fp:
                pickle.dump({'X':X_train}, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_sun397(self):
        # Test SUN397 scene vectors
        print('Testing PAM - SUN 397')

        for i in range(10):
            print('Testing_{:0>2d}.txt'.format(i+1))

            path_test = os.path.join(self.config['path'], 'data', 'SUNPartitions', 'Testing_{:0>2d}.txt'.format(i+1))
            root = os.path.join(self.config['path'], 'data', 'SUN397')
            length = len(open(path_test).readlines())

            X_test = []

            with open(path_test, 'r', encoding='ISO-8859-1') as archive:
                for idx, line in enumerate(archive):
                    try:
                        sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                        path = root + line.strip()

                        ## YOLO

                        im_fname = utils.download('', path = path)
                        img = Image.open(im_fname)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(im_fname)
                        x, img = data.transforms.presets.rcnn.load_test(im_fname)

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
                        
                        ids_bool = list([0] * len(self.coco_classes))

                        for i in ids:
                            ids_bool[i] = 1

                        ## ImageNet

                        img = self.img_channels(img)
                        img = Image.fromarray(np.uint8(img))

                        vec = []
                        for b in boxes:
                            x1 = b[0]
                            y1 = b[1]
                            x2 = b[2]
                            y2 = b[3]

                            box = (x1, y1, x2, y2)
                            region = img.crop(box)
                            vec.append(self.img2vec.get_vec(region).tolist())
                        vec = [float(sum(l))/len(l) for l in zip(*vec)]

                        if len(vec) == 0:
                            vec = np.zeros(self.img2vec.layer_output_size_pool).tolist()

                        ## Join vectors

                        join_vec = ids_bool + list(vec)

                        X_test.append(join_vec)
                    except Exception as e:
                        with open(os.path.join(self.config['path'], 'error_testing_{:0>2d}.log'.format(i+1)), 'a') as log_file:
                            log_file.write('Error at {} \n'.format(line))
                            log_file.write(str(e) + '\n')
                            log_file.write('@@@\n')
                        pass
            print('')

            with open(self.test_sun397_path_pam + '_testing_{:0>2d}'.format(i+1), 'wb') as fp:
                pickle.dump({'X':X_test}, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def train_indoor(self):
        # Train PAM of MIT Indoor 67
        print('Training PAM - MIT Indoor 67')

        X_train = []

        path_train = os.path.join(self.config['path'], 'data', 'TrainImages.txt')

        length = len(open(path_train).readlines())

        with open(path_train, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    ## YOLO

                    im_fname = utils.download('', path = path)
                    x, img = data.transforms.presets.rcnn.load_test(im_fname)

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
                    
                    ids_bool = list([0] * len(self.coco_classes))

                    for i in ids:
                        ids_bool[i] = 1

                    ## ImageNet

                    img = self.img_channels(img)
                    img = Image.fromarray(np.uint8(img))

                    vec = []
                    for b in boxes:
                        x1 = b[0]
                        y1 = b[1]
                        x2 = b[2]
                        y2 = b[3]

                        box = (x1, y1, x2, y2)
                        region = img.crop(box)
                        vec.append(self.img2vec.get_vec(region).tolist())
                    vec = [float(sum(l))/len(l) for l in zip(*vec)]

                    if len(vec) == 0:
                        vec = np.zeros(self.img2vec.layer_output_size_pool).tolist()

                    ## Join vectors

                    join_vec = ids_bool + list(vec)

                    X_train.append(join_vec)
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return
                    
        self.train_data = {'X':X_train}
        with open(self.train_indoor_path_pam, 'wb') as fp:
            pickle.dump(self.train_data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def test_indoor(self):
        # Test PAM on MIT Indoor 67
        print('Testing PAM - MIT Indoor 67')

        X_test = []

        path_test = os.path.join(self.config['path'], 'data', 'TestImages.txt')

        length = len(open(path_test).readlines())
        with open(path_test, 'r', encoding='ISO-8859-1') as archive:
            for idx, line in enumerate(archive):
                try:
                    sys.stdout.write('Reading... ' + str(idx+1) + '/' + str(length) + '\r')

                    path = os.path.join(self.config['path'], 'data', 'MITImages', line.strip())

                    ## YOLO

                    im_fname = utils.download('', path = path)
                    x, img = data.transforms.presets.rcnn.load_test(im_fname)

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
                    
                    ids_bool = list([0] * len(self.coco_classes))

                    for i in ids:
                        ids_bool[i] = 1

                    ## ImageNet

                    img = self.img_channels(img)
                    img = Image.fromarray(np.uint8(img))

                    vec = []
                    for b in boxes:
                        x1 = b[0]
                        y1 = b[1]
                        x2 = b[2]
                        y2 = b[3]

                        box = (x1, y1, x2, y2)
                        region = img.crop(box)
                        vec.append(self.img2vec.get_vec(region).tolist())
                    vec = [float(sum(l))/len(l) for l in zip(*vec)]

                    if len(vec) == 0:
                        vec = np.zeros(self.img2vec.layer_output_size_pool).tolist()

                    ## Join vectors

                    join_vec = ids_bool + list(vec)

                    X_test.append(join_vec)
                except Exception as e:
                    print('Error at {}'.format(line))
                    print(str(e))
                    return

        self.test_data = {'X':X_test}
        with open(self.test_indoor_path_pam, 'wb') as fp:
            pickle.dump(self.test_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
