from operator import index
from aexad.tools.create_dataset import square, square_diff, mvtec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kornia.filters import gaussian_blur2d
import torch

def weighted_htmaps(htmaps, n_pixels=1):
    w_htmaps = np.empty_like(htmaps)
    for k in range(len(htmaps)):
        ht = htmaps[k]
        for i in range(ht.shape[0]):
            for j in range(ht.shape[1]):
                idx_i = np.arange(i - n_pixels, i + n_pixels + 1)
                idx_i = idx_i[np.where((idx_i >= 0) & (idx_i < ht.shape[-2]))[0]]
                idx_j = np.arange(j - n_pixels, j + n_pixels + 1)
                idx_j = idx_j[np.where((idx_j >= 0) & (idx_j < ht.shape[-2]))[0]]
                s = ht[idx_i][:, idx_j].sum() - ht[i, j]
                w_htmaps[k, i, j] = ht[i, j] * (s / (len(idx_i) * len(idx_j) - 1))
    return w_htmaps


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

def plot_results(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))

    if method == 'aexad_conv':
        idx = np.argsort(scores_aexad[Y_test==0])[::-1]

    plt.title(f'{method}')
    #print(idx[:5])
    num=2
    examples=1
    for i in range(examples):
        plt.subplot(examples, num, (i*num)+1)
        print(htmaps_aexad.shape)
        plot_image(X_test[Y_test==0][idx[i]])

        plt.subplot(examples, num, (i*num) + 2)
        plot_heatmap(htmaps_aexad[Y_test == 0][idx[i]])

    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_results_anom(path, method):

    if  method == 'aaexad':
        idx = np.argsort(scores_aexad[Y_test == 1])

    plt.title(f'{method}')

    subplot_per_row = 3
    examples=5
    for i in range(examples):
        plt.subplot(5, subplot_per_row, (i*subplot_per_row)+1)
        plot_image(X_test[Y_test==1][idx[i]])

        plt.subplot(examples, subplot_per_row, (i*subplot_per_row)+2)

        plt.title('AEXAD')
        plot_heatmap(htmaps_aexad[Y_test==1][idx[i]])
        plt.subplot(examples, subplot_per_row, (i*subplot_per_row)+3)

        plot_heatmap(GT_test[Y_test==1][idx[i]])

    plt.show()

def plot_results_anom_top(path, method):

    if method == 'aaexad':
        idx = np.argsort(scores_aexad[Y_test == 1])[::-1]

    plt.title(f'{method}')

    # Apply Gaussian blur to heatmaps
    htmaps_aexad_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad), kernel_size=(15, 15), sigma=(4, 4))

    # Set number of examples to show
    examples = 2
    for i in range(examples):
        image = X_test[Y_test == 1][idx[i]]
        image = image.transpose(2,0,1)
        plot_image(image)
        plt.show()
        plot_heatmap(htmaps_aexad_f[Y_test == 1][idx[i]])
        plt.show()
        gt=GT_test[Y_test == 1][idx[i]]
        gt = gt.transpose(2,0,1)
        plot_heatmap(gt)
        plt.show()

ret_path = "results/output/mvtec"
data = "results/test_data/mvtec"
GT_test = np.load(open(os.path.join(ret_path, 'gt.npy'), 'rb'))
Y_test = np.load(open(os.path.join(ret_path, 'labels.npy'), 'rb'))
htmaps_aexad = np.load(open(os.path.join(ret_path, 'aexad_htmaps_f.npy'), 'rb'))
scores_aexad = np.load(open(os.path.join(ret_path, 'aexad_scores_f.npy'), 'rb'))
X_test=np.load(open(os.path.join(data, 'X_test.npy'), 'rb'))
plot_results_anom_top(ret_path,'aaexad')