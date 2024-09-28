import numpy as np
import matplotlib.pyplot as plt
import os

from aexad.tools.create_dataset import square, square_diff


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
    plt.show()


def plot_heatmap(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    if colorbar:
        plt.colorbar()
    plt.show()

def plot_image_tosave(image, cmap='gray'):
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
    plt.axis('off')
# plt.show()


def plot_heatmap_tosave(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    plt.axis('off')
    '''if colorbar:
        plt.colorbar()'''
    #plt.show()

def plot_results(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    #perc_anom_test = np.sum(Y_test) / len(Y_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==0])[::-1]
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==0])[::-1]
    elif method == 'ae':
       idx = np.argsort(scores_ae[Y_test==0])[::-1]
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test==0])[::-1]

    plt.title(f'{method}')
    print(idx[:5])

    for i in range(5):
        plt.subplot(5, 5, (i*5)+1)
        #print(htmaps_aexad.shape)
        plot_image(X_test[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5)+2)
        plot_heatmap(htmaps_dev[Y_test==0][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 5, (i*5)+3)
        plot_heatmap(htmaps_fcdd[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5)+4)
        plot_heatmap(htmaps_ae[Y_test==0][idx[i]])
        plt.subplot(5, 5, (i*5) + 5)
        plot_heatmap(htmaps_aexad_conv[Y_test == 0][idx[i]])

    plt.show()

def plot_results_anom(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test_file = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    perc_anom_test = np.sum(Y_test_file) / len(Y_test_file)
    print(perc_anom_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    print('TEST: ', X_test[0,0,0,0])

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    #htmaps_aexad_lenet = np.load(open(os.path.join(path.replace('mnist', 'conv/mnist'), 'aexad_htmaps_lenet.npy'), 'rb'))
    #scores_aexad_lenet = np.load(open(os.path.join(path.replace('mnist', 'conv/mnist'), 'aexad_scores_lenet.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores_conv.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==1])
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==1])
    elif method == 'ae':
        idx = np.argsort(scores_ae[Y_test==1])
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test == 1])


    plt.title(f'{method}')
    print(idx[:5])

    for i in range(5):
        plt.subplot(5, 6, (i*6)+1)
        plot_image(X_test[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+2)
        plt.title('DEV')
        plot_heatmap(htmaps_dev[Y_test==1][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 6, (i*6)+3)
        plt.title('FCDD')
        plot_heatmap(htmaps_fcdd[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+4)
        plt.title('AE')
        plot_heatmap(htmaps_ae[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+5)
        plt.title('AEXAD')
        plot_heatmap(htmaps_aexad_conv[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+6)
        #intensity = GT_test[Y_test==1][idx[i]]*X_test[Y_test==1][idx[i]]
        #plt.title(np.unique(intensity).sum())
        plot_heatmap(GT_test[Y_test==1][idx[i]])

    plt.show()

def plot_results_anom_top(path, method):
    infos = (os.path.normpath(path)).split(os.sep)
    dataset = infos[-3]
    c = int(infos[-2])
    seed = int(infos[-1])

    Y_test_file = np.load(open(os.path.join(path, 'labels.npy'), 'rb'))
    perc_anom_test = np.sum(Y_test_file) / len(Y_test_file)
    print(perc_anom_test)

    _, _, X_test, Y_test, _, GT_test = square_diff(c, perc_anom_train=0.02, perc_anom_test=0.10, size=5,
               intensity=0.25, DATASET=dataset, seed=seed)

    htmaps_dev = np.load(open(os.path.join(path, 'deviation_htmaps.npy'), 'rb'))
    scores_dev = np.load(open(os.path.join(path, 'deviation_scores.npy'), 'rb'))
    htmaps_fcdd = np.load(open(os.path.join(path, 'fcdd_htmaps.npy'), 'rb'))
    scores_fcdd = np.load(open(os.path.join(path, 'fcdd_scores.npy'), 'rb'))
    htmaps_ae = np.load(open(os.path.join(path, 'ae_htmaps.npy'), 'rb'))
    scores_ae = np.load(open(os.path.join(path, 'ae_scores.npy'), 'rb'))
    htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps_conv.npy'), 'rb'))
    scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores_conv.npy'), 'rb'))

    if method == 'deviation':
        idx = np.argsort(scores_dev[Y_test==1])[::-1]
    elif method == 'fcdd':
        idx = np.argsort(scores_fcdd[Y_test==1])[::-1]
    elif method == 'ae':
        idx = np.argsort(scores_ae[Y_test==1])[::-1]
    elif method == 'aexad_conv':
        idx = np.argsort(scores_aexad_conv[Y_test == 1])[::-1]

    print(idx[:5])


    plt.title(f'{method}')
    for i in range(5):
        plt.subplot(5, 6, (i*6)+1)
        plot_image(X_test[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+2)
        plot_heatmap(htmaps_dev[Y_test==1][idx[i]].reshape(1, X_test.shape[-2], X_test.shape[-1]))
        plt.subplot(5, 6, (i*6)+3)
        plot_heatmap(htmaps_fcdd[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+4)
        plot_heatmap(htmaps_ae[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+5)
        plot_heatmap(htmaps_aexad_conv[Y_test==1][idx[i]])
        plt.subplot(5, 6, (i*6)+6)
        #plot_image(np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]])
        #print((np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]]).max())
        #plt.colorbar()
        #plt.subplot(5, 8, (i*8)+7)
        #plot_heatmap((np.sqrt(htmaps_aexad_conv[Y_test == 1][idx[i]])+X_test[Y_test==1][idx[i]])-X_test[Y_test==1][idx[i]])
        #plt.subplot(5, 8, (i*8)+8)
        intensity = GT_test[Y_test==1][idx[i]]*X_test[Y_test==1][idx[i]]
        plt.title(np.unique(intensity).sum())
        plot_heatmap(GT_test[Y_test==1][idx[i]])

    plt.show()

def plot_results(X_test, Y_test, GT_test, htmaps, ids):
    plt.figure(figsize=(9,15))
    for i in range(5):
        plt.subplot(5, 3, (i * 3) + 1)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(X_test[ids[i]])
        plt.subplot(5, 3, (i * 3) + 2)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(htmaps[ids[i]][0], cmap='jet', vmin=htmaps[ids[i]][0].min(), vmax=htmaps[ids[i]][0].max())
        #plt.colorbar()
        plt.subplot(5, 3, (i * 3) + 3)
        plt.tight_layout()
        plt.axis(False)
        plt.imshow(GT_test[ids[i]][0], cmap='jet')
    plt.show()