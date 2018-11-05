# Get MIT Indoor 67 annotation data
## Henrique X. Goulart

import os
import xmltodict
import json
from fnmatch import fnmatch

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

ann_path = os.path.join(config['path'], 'data', 'indoor_ann.json')

if os.path.exists(ann_path):
    print('Loading annotations data')
else:
    print('Annotations data not found!')

    ann_dir = os.path.join(config['path'], 'data', 'MITAnnotations')
    pattern = '*.xml'

    annotations = {}

    for path, subdirs, files in os.walk(ann_dir):
        length = len(files)

        for idx, name in enumerate(files):
            if fnmatch(name, pattern):
                try:
                    doc = {}
                    with open(os.path.join(path, name)) as fs:
                        doc = xmltodict.parse(fs.read())

                    img_name = os.path.splitext(name)[0] # Remove extension
                    annotations[img_name] = doc
                except:
                    print('Error at {}'.format(os.path.join(path, name)))

    print(annotations['hut4'])

    with open(ann_path, 'w') as fp:
        json.dump(annotations, fp, sort_keys=True, indent=4)
