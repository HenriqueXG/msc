import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from src.img_to_vec import Img2Vec

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

img2vec = Img2Vec(model = 'resnet-18-Places')

img_name = 'sun_ajlaxptrakzafroo'
img_path = os.path.join(config['path'], 'data/SUN397/a/aqueduct', img_name) + '.jpg'
img = Image.open(img_path)

# plt.imshow(img)

vec = img2vec.get_vec(img)

rel_parts = Path(path).relative_to(root).parts[1:]
scene_class = Path(*list(rel_parts)).as_posix()
