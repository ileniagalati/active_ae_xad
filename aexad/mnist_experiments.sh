#!/bin/bash

for i in {0..9}
do
  python3 launch_experiments.py -ds mnist -c $i -patr 0.02 -pate 0.1 -s 29 -size 5 -i rand
done