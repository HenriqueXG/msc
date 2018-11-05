# Setup project for compilation
## Henrique X. Goulart

import sys
import json
import os
import multiprocessing
from setuptools import setup
from Cython.Build import cythonize
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext

NB_COMPILE_JOBS = multiprocessing.cpu_count()

ext_modules = [
    Extension('graph', ['src/graph.py']),
    Extension('img_to_vec', ['src/img_to_vec.py']),
    Extension('declarative', ['src/declarative.py'])
]

def setup_given_extensions(extensions):
    setup(
        name = 'msc',
        cmdclass = {'build_ext': build_ext},
        ext_modules = cythonize(extensions)
    )

def setup_extensions_in_sequential():
    setup_given_extensions(ext_modules)

def setup_extensions_in_parallel():
    cythonize(ext_modules, nthreads=NB_COMPILE_JOBS)
    pool = multiprocessing.Pool(processes=NB_COMPILE_JOBS)
    pool.map(setup_given_extensions, ext_modules)
    pool.close()
    pool.join()

if 'build_ext' in sys.argv:
    print ('Compiling in parallel...')
    setup_extensions_in_parallel()
else:
    print ('Compiling in sequential...')
    setup_extensions_in_sequential()

with open('config.json', 'r') as fp:
    self.config = json.load(fp)

os.system('python3 ' + os.path.join(config['path'], 'utils', 'sun_obj_annotations.py'))
