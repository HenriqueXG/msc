#!/bin/sh

# Compile project

rm lib/*.so
rm src/*.c
rm main
rm main.c

python3 setup.py build_ext --inplace
python3 rename.py
python3 src/sun_obj_annotations.py
mv *.so lib/

cython --embed -o main.c main.py

cmake .
make -j4
