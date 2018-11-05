# Rename compiled libs
## Henrique X. Goulart

import os
from fnmatch import fnmatch
import json

if __name__ == '__main__':
    config = {}
    with open('config.json', 'r') as fp:
        config = json.load(fp)

    directory =  os.path.join(config['path'], 'lib')
    pattern = '*.so'

    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch(name, pattern):
                pos = name.find('.')
                new_name = name[:pos]
                new_name += '.so'
                os.rename(os.path.join(path, name), new_name)
