# Msc.: CLASSIFICAÇÃO DE CENAS EM IMAGENS ATRAVÉS DA ARQUITETURA COGNITIVA LIDA E BAG OF FEATURES

## Installing

Install pip (Python 3) dependencies:

```pip install -r requirements.txt```

Install CMake and g++.

Download the pre-trained Places-CNN: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar . Then, move it to ```data``` folder.

Ajust the correct path of this project in your computer by setting the value of ```path``` in ```config.json```.

## Datasets

### MIT Indoor 67

Download the dataset from: http://web.mit.edu/torralba/www/indoor.html

Extract the image folder content into ```data/MITImages```

Move TrainImages.txt to ```data```

Move TestImages.txt to ```data```

### SUN 397

To do

## Run

To run with intepreter: ```python3 main.py```

To run with Cython: ```./configure && ./main```

## Misc

Once trained, the train files (Memories) is saved into ```data```. 

The execution takes ~6Gb of RAM.
