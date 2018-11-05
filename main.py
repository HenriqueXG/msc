## Henrique X. Goulart

import json
import sys
import os
import re

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'])

try:
    from lib.graph import Graph
    from lib.declarative import Declarative
except ImportError:
    from src.graph import Graph
    from src.declarative import Declarative

if __name__ == '__main__':
    graph = Graph(arch = config['arch_obj'])
    declarative = Declarative(graph)

    declarative.test_scene_indoor()
