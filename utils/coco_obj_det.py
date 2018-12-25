from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import json
import os
from mxnet.gluon.data.vision import transforms
import mxnet as mx
from mxnet import image
import numpy as np
from matplotlib import patches
from mxnet import nd, image
from gluoncv.data.transforms.presets.imagenet import transform_eval

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

np.random.seed(1)

input_path = os.path.join(config['path'], 'debug', 'test_images')
pic_name = 'highway.jpg'

# you can change it to your image filename
filename = os.path.join(input_path, pic_name)

net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

im_fname = utils.download('', path=filename)
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

box_ids, sscores, bboxes = net(x)

boxes = []
scores = []
for i in range(len(sscores[0])):
    if sscores[0][i][0] > 0.7:
        boxes.append(bboxes.asnumpy()[0][i])
        scores.append(float(sscores.asnumpy()[0][i][0]))

ax = utils.viz.plot_bbox(orig_img, bboxes[0], sscores[0], box_ids[0], class_names=net.classes)
ax.axis('off')

fig, ax = plt.subplots()
for s, b in zip(scores, boxes):
    x1 = b[0]
    y1 = b[1]
    x2 = b[2]
    y2 = b[3]

    pos = [(x1, y1), (x2, y2)]
    dx = x2 - x1
    dy = -(y2 - y1)

    color = np.random.rand(3)
    rect = patches.Rectangle((x1, y2), dx, dy, linewidth = 3, edgecolor = [0,0,1], facecolor = 'None')
    ax.add_patch(rect)

ax.imshow(orig_img)
ax.axis('off')

plt.show()