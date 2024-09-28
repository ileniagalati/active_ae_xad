#!/bin/bash

for i in {0..9}
do
  python test_ae.py -ds fmnist -c $i -patr 0.02 -pate 0.1 -s 75 -size 5 -i rand --no-cuda
done

