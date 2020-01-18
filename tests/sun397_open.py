import json
import os
import sys
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

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
path = root + '/a/assembly_line/sun_aohzzprghfrqbgkt.jpg'

img = Image.open(path)
img = img_channels(img)
vec = img2vec.get_vec(img)