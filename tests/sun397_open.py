import json
import os
import sys
import pickle
import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from gluoncv import model_zoo, data, utils

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(os.path.join(config['path'], 'src'))
from img_to_vec import Img2Vec
img2vec = Img2Vec(model = config['arch_scene'])

def img_channels(img):
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

root = os.path.join(config['path'], 'data', 'SUN397')
path = root + '/a/assembly_line/sun_ajckcfldgdrdjogj.jpg'
path_train = os.path.join(config['path'], 'data', 'SUNPartitions', 'Training_{:0>2d}.txt'.format(config['sun397_it']))

## YOLO

im_fname = utils.download('', path = path)
img = Image.open(im_fname)
if img.mode != 'RGB':
    img = img.convert('RGB')
    img.save(im_fname)
x, img = data.transforms.presets.rcnn.load_test(im_fname)


with open(path_train, 'r', encoding='ISO-8859-1') as archive:
    for idx, line in enumerate(archive):
        scene_class = line.split('/')[-2]
        print(scene_class)
