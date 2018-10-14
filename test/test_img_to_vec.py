import sys
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import json

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'] + '/src')
from img_to_vec import Img2Vec

input_path = config['path'] + '/test/test_images'

img2vec = Img2Vec(model='resnet-18')

# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    pics[filename] = vec

pic_name = 'face.jpg'
# pic_name = str(input("Which filename would you like similarities for?\n"))

sims = {}
for key in list(pics.keys()):
    if key == pic_name:
        continue

    sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

d_view = [(v, k) for k, v in sims.items()]
d_view.sort(reverse=True)
for v, k in d_view:
    print(v, k)
