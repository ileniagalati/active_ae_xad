import argparse
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import subprocess
import torch
from PIL import Image

from aexad.tools.create_dataset import load_brainMRI_dataset
from aexad.tools.create_dataset import mvtec
from active_aexad_script import launch as launch_aexad
import numpy as np
from aexad.tools.utils import plot_image_tosave, plot_heatmap_tosave
from active_utils import *

import argparse
import logging

def plot_image(image, cmap='gray'):
    '''
    Method to plot an image
    :param image: (C x H x W) channel-first-format image
    :return:
    '''
    if image.shape[0] == 1:
        image = image[0, :, :]
    else:
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    #plt.show()

def plot_heatmap(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    plt.axis("off")
    #plt.show()


if __name__ == '__main__':
    path = "mvtec_results/leather/b_split/500ep_decay_lr_20ep_iteration/weights/1.0/29"
    log_path = os.path.join(path, "logs", "0")

    if not os.path.exists(os.path.join(path, "prova_out_plot_an")):
        os.makedirs(os.path.join(path, "prova_out_plot_an"))

    if not os.path.exists(os.path.join(path, "prova_out_plot_norm")):
        os.makedirs(os.path.join(path, "prova_out_plot_norm"))


    Y = np.load(open(os.path.join(path, "output", "labels.npy"), 'rb'))
    scores = np.load(open(os.path.join(log_path, f'aexad_scores_0.npy'), 'rb'))
    htmaps = np.load(open(os.path.join(log_path, "aexad_htmaps_0.npy"), "rb"))

    idx_anom = np.argsort(scores[Y == 1])[:10]
    idx_norm= np.argsort(scores[Y == 0])[-10:][::-1]

    X_test = np.load(open(os.path.join(os.path.join(path, "test_data"), str(1), 'X_test.npy'), 'rb'))
    O = np.load(open(os.path.join(os.path.join(path, "logs"), str(0), f'output_0.npy'), 'rb'))
    print(len(X_test))

    for x in range(0, 10):
        image = X_test[Y == 1][idx_anom[x]]
        image = image.transpose(2, 0, 1)
        plot_image(image)
        plt.savefig(os.path.join(path, "prova_out_plot_an",  f"img_0_{x}.png"))

        out = O[Y == 1][idx_anom[x]]
        plot_image(out)
        plt.savefig(os.path.join(path, "prova_out_plot_an", f"out_0_{x}.png"))

        plot_heatmap(htmaps[Y == 1][idx_anom[x]])
        plt.savefig(os.path.join(path, "prova_out_plot_an", f"ht_0_{x}.png"))

    for x in range(0, 10):
        image = X_test[Y == 0][idx_norm[x]]
        image = image.transpose(2, 0, 1)
        plot_image(image)
        plt.savefig(os.path.join(path, "prova_out_plot_norm",  f"img_0_{x}.png"))

        out = O[Y == 0][idx_norm[x]]
        plot_image(out)
        plt.savefig(os.path.join(path, "prova_out_plot_norm", f"out_0_{x}.png"))

        plot_heatmap(htmaps[Y == 0][idx_norm[x]])
        plt.savefig(os.path.join(path, "prova_out_plot_norm", f"ht_0_{x}.png"))



