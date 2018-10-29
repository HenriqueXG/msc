# Get sun annotation data

import os
import xmltodict
import json

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

if os.path.exists(os.path.join(config['path'], 'data/sun2012_ann.json')):
    pass
else:
    print('Annotations data not found!')

    root_dir = os.path.join(config['path'], 'data/SUN2012pascalformat')
    img_dir = os.path.join(root_dir, 'JPEGImages')
    ann_dir = os.path.join(root_dir, 'Annotations')
    set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

    annotations = {}
    for f in os.listdir(ann_dir):
        try:
            doc = {}
            with open(os.path.join(ann_dir, f)) as fs:
                doc = xmltodict.parse(fs.read())

            name = doc['annotation']['filename'].replace('.jpg','')
            annotations[name] = doc
        except:
            print(f)

    with open(os.path.join(config['path'], 'data/sun2012_ann.json'), 'w') as fp:
        json.dump(annotations, fp, sort_keys=True, indent=4)
