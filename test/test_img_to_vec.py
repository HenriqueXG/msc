import sys
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import json
from torch.nn import functional as F
from torchvision import transforms as trn
from torch.autograd import Variable as V

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'] + '/src')
from img_to_vec import Img2Vec

input_path = config['path'] + '/test/test_images'

arch = 'resnet-50'
img2vec = Img2Vec(model=arch)

# For each test image, we store the filename and vector as key, value in a dictionary
pics = {}
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    pics[filename] = vec

pic_name = 'cat.jpg'
# pic_name = str(input("Which filename would you like similarities for?\n"))

####################################### SIMILARITY #######################################

sims = {}
for key in list(pics.keys()):
    if key == pic_name:
        continue

    sims[key] = cosine_similarity(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

print('{} similarity on {}'.format(arch, pic_name))
d_view = [(v, k) for k, v in sims.items()]
d_view.sort(reverse=True)
for v, k in d_view:
    print(v, k)
print('-------------------------------')

####################################### PREDICTION #######################################
if arch.find('Places') == -1:
    class_path = config['path'] + '/data/imagenet1000_clsid_to_human.txt'

    classes = list()
    with open(class_path) as class_file:
        for line in class_file:
            if line[-1] == ',':
                line[-1] = ''
            l = line.replace('{','').replace('}','').replace('\'','').strip().strip(',').split(':')
            l[1] = l[1].strip()
            l[0] = int(l[0])
            classes.append(l)
    classes = tuple(classes)

    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # input_img = img2vec.normalize(img2vec.to_tensor(img2vec.scaler(Image.open(os.path.join(input_path, pic_name))))).unsqueeze(0).to(img2vec.device)
    input_img = V(centre_crop(Image.open(os.path.join(input_path, pic_name))).unsqueeze(0))

    # forward pass
    logit = img2vec.model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{} prediction on {}'.format(arch, pic_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
