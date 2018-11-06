import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os

class Img2Vec():
    def __init__(self, cuda=False, model='resnet-18-ImageNet', layer='default'):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        """

        self.config = {}
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)

        self.layer_output_size_pool = 1

        self.device = torch.device("cuda" if cuda else "cpu")
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size_pool, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: name ofS layer
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18-ImageNet':
            model = models.resnet18(pretrained=True)

            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size_pool = 512
                self.layer_output_size = 1
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-50-ImageNet':
            model = models.resnet50(pretrained=True)

            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size_pool = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-18-Places':
            model = models.resnet18()

            file = torch.load(os.path.join(self.config['path'], 'data', 'resnet18_places365.pth.tar'), map_location='cpu')['state_dict']

            new_model = {}# rename key
            for key, value in file.items():
                new_key = key.replace('module.','')
                new_model[new_key] = value
            file = new_model

            model.fc = torch.nn.Linear(in_features=512, out_features=365, bias=True) # reshape for old resnet18 model

            model.load_state_dict(file)


            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size_pool = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-50-Places':
            model = models.resnet50()

            file = torch.load(os.path.join(self.config['path'], 'data', 'resnet50_places365.pth.tar'), map_location = 'cpu')['state_dict']

            new_model = {}# rename key
            for key, value in file.items():
                new_key = key.replace('module.','')
                new_model[new_key] = value
            file = new_model

            model.fc = torch.nn.Linear(in_features=2048, out_features=365, bias=True) # reshape for old resnet50 model

            model.load_state_dict(file)

            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size_pool = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)
