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
except ImportError:
    from src.graph import Graph

if __name__ == '__main__':
    graph = Graph(arch = config['arch_obj'])
