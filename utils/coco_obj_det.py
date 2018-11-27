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
pic_name = 'patio.jpg'

ctx = mx.cpu(0)

img = image.imread(os.path.join(input_path, pic_name))

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
img = transform_fn(img)
img = img.expand_dims(0).as_in_context(ctx)

#model = gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True)
model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
#model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True)

output = model.demo(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'ade20k')

plt.imshow(mask)
plt.show()
