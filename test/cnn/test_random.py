import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import sys
import json
from torch.nn import functional as F
from torchvision import transforms as trn
from PIL import Image
import os
from torch.autograd import Variable as V
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'] + '/src')

from simple_cnn import SimpleCNN

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(os.path.join(config['path'] + '/test/test_images/', 'cat.jpg'))
input_img = V(transform(img).unsqueeze(0))

train_set = torchvision.datasets.CIFAR10(root=config['path'] + '/data', train=True, download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root=config['path'] + '/data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

#Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

#Test
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

#Test and validation loaders have constant batch sizes, so we can define them directly
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

CNN = SimpleCNN()
# CNN.trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)

input_path = config['path'] + '/test/test_images'
pics = {}
for file in os.listdir(input_path):
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = CNN.forward(V(transform(img).unsqueeze(0)))
    vec = vec.detach().numpy()[0, :]
    pics[filename] = vec

pic_name = 'cat.jpg'

sims = {}
for key in list(pics.keys()):
    if key == pic_name:
        continue

    sims[key] = euclidean_distances(pics[pic_name].reshape((1, -1)), pics[key].reshape((1, -1)))[0][0]

print('{} similarity on {}'.format('Random CNN', pic_name))
d_view = [(v, k) for k, v in sims.items()]
d_view.sort(reverse=True)
for v, k in d_view:
    print(v, k)
