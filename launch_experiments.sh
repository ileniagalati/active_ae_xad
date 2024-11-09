#!/bin/bash

ds="datasets/mvtec"
ret="mvtec_results/"

! python3 uniform_active_launch.py -ds $ds -b 2 -e 1 -s 29 -p 1 -l 0 -r $ret