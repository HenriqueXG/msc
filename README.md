# MSc.: CLASSIFICAÇÃO DE CENAS EM IMAGENS ATRAVÉS DA ARQUITETURA COGNITIVA LIDA

  

## Installing

  

Install pip (Python 3) dependencies:

  

```pip install -r requirements.txt```

  

Install Make, CMake, python3x-devel and g++ (gcc).

  

Download the pre-trained Places-CNN: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar . Then, move it to ```data``` folder.

  

Ajust the correct path of this project in your computer by setting the value of ```path``` in ```config.json```.

  

## Datasets

  

### MIT Indoor 67

  

Download the dataset from: http://web.mit.edu/torralba/www/indoor.html

  

Extract the image folder content into ```data/MITImages```

  

Move TrainImages.txt to ```data```

  

Move TestImages.txt to ```data```

  

### SUN 397

  

Download the dataset from: https://vision.cs.princeton.edu/projects/2010/SUN/

  

Extract the image folder content into ```data/SUN397```

  

Extract the partitions folder content into ```data/SUNPartitions```

  

## Run

  

Configure ```config.json``` file:
```jsonc
{
"path":"{path/to/msc}", 
"arch_pam":"yolo3_darknet53_coco", (YOLO architecture)
"arch_scene":"resnet-50-Places", (Places-CNN)
"arch_obj":"resnet-18-ImageNet", (ImageNet-CNN)
"dataset":"{indoor||sun397}", (select dataset)
"sun397_it": 1, (SUN 397 partition id: 1-10)
"alpha":0.5, (alpha value)
"pam_threshold":0.5, (YOLO threshold)
"hidden_units": 1000, (number of hidden units in MLP)
"activation": "{logistic||relu||tanh}", (select MLP activation function)
"optimizer": "{adam||sgd}", (select MLP optimization function)
"kernel": "rbf", (select SVM kernel)
"it": 1, (number of evaluations)
"exp_a": 0, (bool for run Experiment A)
"exp_b": 1 (bool for run Experiment B)
}
```

  

To run with intepreter: ```python3 main.py```

  

To run with Cython: ```./configure && ./main```

  

## Misc

  

Once trained, the train files (Memories) is saved into ```data```.

  

The execution takes ~6Gb of RAM for MIT Indoor 37 and ~20Gb for SUN 397.

  

Running in background: ```rm -rf /tmp/msc.log && nohup python3 -u main.py >> /tmp/msc.log 2>&1 &```.
