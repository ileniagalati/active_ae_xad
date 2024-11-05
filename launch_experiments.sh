#!/bin/bash

ds="datasets/mvtec"
ret="mvtec_results/"

! python3 pure_active_launch.py -ds $ds -b 1 -e 1 -s 2 -p 0.75 -l 0 -r $ret