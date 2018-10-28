import json
import sys

config = {}
with open('config.json', 'r') as fp:
    config = json.load(fp)

sys.path.append(config['path'] + '/src')
from graph import Graph

graph = Graph()
