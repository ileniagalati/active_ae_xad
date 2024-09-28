#/content/drive/MyDrive/AE_XAD/launch_experiments.py
import argparse
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import requests
import competitors

import tools

from tools.create_dataset import square, square_diff, mvtec
from aexad_script import launch as launch_aexad
import plot_competitors_script as plot
import warnings
warnings.filterwarnings("ignore")

ds="mnist"
c=0
patr=0.02
pate=0.1
s=29
size=5
i='rand'
print("here")
print(torch.cuda.is_available())
for c in range (10):
    print(f'Class: {c}\n')
    if i != 'rand':
        i = float(i)

    if ds == 'mnist'  or ds == 'fmnist':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(c, perc_anom_train=patr, perc_anom_test=pate, size=size,
                   intensity=i, DATASET=ds, seed=s)
    elif ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(c, perc_anom_train=patr, perc_anom_è_test=pate, size=size,
                        intensity=i, DATASET=dataset, seed=s)

    data_path = os.path.join('test_data', ds, str(c), str(s))

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    ret_path = os.path.join('results', ds, str(c), str(s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    #pickle.dump(open(os.path.join(ret_path, 'args'), 'wb'))
    with open(os.path.join(ret_path, 'variables.pickle'), 'wb') as f:
        pickle.dump({
            'dataset': ds,
            'considered_class': c,
            'patr': patr,
            'pate': pate,
            'modification_intensity': i,
            'seed': s,
            'size': size,
            'ret_path': ret_path
        }, f)

    times = []

    def f(x):
        return 1-x
    lambda_p=(28*28) / 25
    lambda_p = torch.tensor(lambda_p)

    heatmaps, scores, _, _, tot_time = launch_aexad(data_path,1, 16, 32,lambda_p, None, f, 'conv',
                                                    save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'wb'), scores)

    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))
    print(times)

    shutil.rmtree(data_path)