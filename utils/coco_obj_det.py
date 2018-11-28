from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import json
import os
from mxnet.gluon.data.vision import transforms
import mxnet as mx
from mxnet import image

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

input_path = os.path.join(config['path'], 'debug', 'test_images')
pic_name = 'catdog.jpg'

# you can change it to your image filename
filename = os.path.join(input_path, pic_name)

net = model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True)

im_fname = utils.download('', path=filename)
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()