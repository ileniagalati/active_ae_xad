#!/bin/bash

ds="datasets/mvtec"
ret="mvtec_results/"

! python3 pure_active_launch.py -ds $ds -b 2 -e 1 -s 2 -p 1 -l 1 -r $ret