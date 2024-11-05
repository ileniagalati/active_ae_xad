from operator import index
from aexad.tools.create_dataset import square, square_diff, mvtec
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline

import matplotlib.pyplot as plt
# Resto del tuo codice


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

    if method == 'aaexad':
        idx = np.argsort(scores_aexad[Y_test == 1])

    plt.title(f'{method}')

    # Apply Gaussian blur to heatmaps
    htmaps_aexad_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad), kernel_size=(15, 15), sigma=(4, 4))

    # Set number of examples to show
    examples = 5
    for i in range(examples):
        image = X_test[Y_test == 1][idx[i]]
        image = image.transpose(2, 0, 1)
        plot_image(image)
        plt.savefig(os.path.join(ret_path, "plot", f"img_{i}.png"))
        plot_heatmap(htmaps_aexad_f[Y_test == 1][idx[i]])
        plt.savefig(os.path.join(ret_path, "plot", f"ht_{i}.png"))
        gt = GT_test[Y_test == 1][idx[i]]
        gt = gt.transpose(2, 0, 1)
        plot_heatmap(gt)
        plt.savefig(os.path.join(ret_path, "plot", f"gt_{i}.png"))

def plot_results_anom_top(path, method):

    if method == 'aaexad':
        idx = np.argsort(scores_aexad[Y_test == 1])[::-1]

    plt.title(f'{method}')

    htmaps_aexad_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad), kernel_size=(15, 15), sigma=(4, 4))

    examples = 5
    for i in range(examples):
        image = X_test[Y_test == 1][idx[i]]
        image = image.transpose(2,0,1)
        plot_image(image)
        plt.savefig(os.path.join(ret_path,"plot",f"img_{i}.png"))
        plot_heatmap(htmaps_aexad_f[Y_test == 1][idx[i]])
        plt.savefig(os.path.join(ret_path,"plot",f"ht_{i}.png"))
        gt=GT_test[Y_test == 1][idx[i]]
        gt = gt.transpose(2,0,1)
        plot_heatmap(gt)
        plt.savefig(os.path.join(ret_path,"plot",f"gt_{i}.png"))


import os
import matplotlib.pyplot as plt


def plot_iteration_results(path, it, model_type):
    for i in range(1,it+1):
        subpath=os.path.join(path, "plot",str(i))
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        if i==it+1:
            i="f"
        GT = np.load(open("mvtec_results/weights/1.0/2/output/gt.npy"), 'rb')
        Y = np.load(open("mvtec_results/weights/1.0/2/output/labels.npy"), 'rb')

        htmaps_aexad = np.load(open(os.path.join(path, str(i), f'aexad_htmaps_{i}.npy'), 'rb'))
        X_test = np.load(open(os.path.join(path, str(i), 'X_test.npy'), 'rb'))

        print (len(X_test))
        print(len(Y))

        for x in range(0, len(X_test)):
            if len(X_test[Y==1]>0):
                image = X_test[Y==1][x]
                image = image.transpose(2, 0, 1)
                plot_image(image)
                plt.savefig(os.path.join(subpath, f"img_{i}_{x}.png"))
                plot_heatmap(htmaps_aexad[Y == 1][x])
                plt.savefig(os.path.join(subpath, f"ht_{i}_{x}.png"))
                gt = GT[Y == 1][x]
                gt = gt.transpose(2, 0, 1)
                plot_heatmap(gt)
                plt.savefig(os.path.join(subpath, f"gt_{i}_{x}.png"))


    plt.show()

plot_iteration_results("mvtec_results/weights/1.0/2/logs", 10,'aaexad')

'''
ret_path = "mvtec_results/weights/0.5/29"
GT_test = np.load(open(os.path.join(ret_path, "output", 'gt.npy'), 'rb'))
Y_test = np.load(open(os.path.join(ret_path, "output", 'labels.npy'), 'rb'))
htmaps_aexad = np.load(open(os.path.join(ret_path, "output",'aexad_htmaps_f.npy'), 'rb'))
scores_aexad = np.load(open(os.path.join(ret_path, "output",'aexad_scores_f.npy'), 'rb'))
X_test=np.load(open(os.path.join(ret_path, "test_data", 'X_test.npy'), 'rb'))

if not os.path.exists(os.path.join(ret_path,"plot")):
    os.makedirs(os.path.join(ret_path,"plot"))
plot_results_anom_top(ret_path,'aaexad')

'''