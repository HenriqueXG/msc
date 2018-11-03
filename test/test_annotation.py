# Test SUN Database annotations
## Henrique X. Goulart

import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

annotations = {}
with open(os.path.join(config['path'], 'data/sun2012_ann.json'), 'r') as fp:
    annotations = json.load(fp)

img_name = 'sun_aiatoxvclffgxjts'
img = Image.open(os.path.join(config['path'], 'data/SUN2012pascalformat/JPEGImages/', img_name) + '.jpg')

fig, ax = plt.subplots(1)

objects = annotations[img_name]['annotation']['object']
for i, obj in enumerate(objects):
    try:
        x1 = int(obj['bndbox']['xmin'])
        x2 = int(obj['bndbox']['xmax'])
        y1 = int(obj['bndbox']['ymin'])
        y2 = int(obj['bndbox']['ymax'])
    except TypeError:
        x1 = int(objects['bndbox']['xmin'])
        x2 = int(objects['bndbox']['xmax'])
        y1 = int(objects['bndbox']['ymin'])
        y2 = int(objects['bndbox']['ymax'])
        pass

    pos = [(x1, y1), (x2, y2)]
    dx = x2 - x1
    dy = y2 - y1
    cx = int(dx/2) + x1
    cy = int(dy/2) + y1

    color = np.random.rand(3)
    rect = patches.Rectangle((x1, y1), dx, dy, linewidth = 1, edgecolor = color, facecolor = 'None')
    ax.add_patch(rect)
    try:
        ax.text(cx, cy, obj['name'], color = color, horizontalalignment = 'center')
    except TypeError:
        ax.text(cx, cy, objects['name'], color = color, horizontalalignment = 'center')
        pass

plt.xticks([]), plt.yticks([])
ax.imshow(img)
plt.show()
