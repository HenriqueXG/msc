# Get sun annotation data
## Henrique X. Goulart

import os
import xmltodict
import json

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

ann_path = os.path.join(config['path'], 'data', 'sun2012_ann.json')

if os.path.exists(ann_path):
    print('Loading annotations data')
else:
    print('Annotations data not found!')

    root_dir = os.path.join(config['path'], 'data', 'SUN2012pascalformat')
    ann_dir = os.path.join(root_dir, 'Annotations')

    annotations = {}
    for f in os.listdir(ann_dir):
        try:
            doc = {}
            with open(os.path.join(ann_dir, f)) as fs:
                doc = xmltodict.parse(fs.read())

            name = doc['annotation']['filename'].replace('.jpg','')
            annotations[name] = doc
        except:
            print('Error at {}'.format(f))

    with open(ann_path, 'w') as fp:
        json.dump(annotations, fp, sort_keys=True, indent=4)
