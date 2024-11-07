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
    Y = np.load(open(os.path.join("mvtec_results/weights/1.0/29", "output", "labels.npy"), 'rb'))

    for i in range(1,it):
        subpath=os.path.join(path, "plot",str(i))
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        if i==it:
            i="f"

        htmaps_aexad = np.load(open(os.path.join(path, str(i), f'aexad_htmaps_{i}.npy'), 'rb'))
        X_test = np.load(open(os.path.join(path, str(i), 'X_test.npy'), 'rb'))

        print (len(X_test))
        print(len(Y))

        for x in range(0, len(X_test[Y==1])):
            if len(X_test[Y==1]>0):
                image = X_test[Y==1][x]
                image = image.transpose(2, 0, 1)
                plot_image(image)
                plt.savefig(os.path.join(subpath, f"img_{i}_{x}.png"))
                plot_heatmap(htmaps_aexad[Y == 1][x])
                plt.savefig(os.path.join(subpath, f"ht_{i}_{x}.png"))

#plot_iteration_results("mvtec_results/weights/1.0/29/logs", 10,'aaexad')

'''
import os
import matplotlib.pyplot as plt
import cv2

def table(folder_path,output_path):
        folder_path = folder_path

        img_files = sorted([f for f in os.listdir(folder_path) if f.startswith('img')])
        ht_files = sorted([f for f in os.listdir(folder_path) if f.startswith('ht')])

        img_list = [cv2.imread(os.path.join(folder_path, f)) for f in img_files]
        ht_list = [cv2.imread(os.path.join(folder_path, f)) for f in ht_files]

        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
        ht_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in ht_list]

        n_cols = max(len(img_list), len(ht_list))

        fig, axs = plt.subplots(2, n_cols, figsize=(2 * n_cols, 9))

        for col in range(n_cols):
            if col < len(img_list):
                axs[0, col].imshow(img_list[col])
                axs[0, col].axis('off')
            if col < len(ht_list):
                axs[1, col].imshow(ht_list[col])
                axs[1, col].axis('off')

        output_path = output_path
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

for i in range(4,10):
    folder_path=f"mvtec_results/weights/1.0/29/logs/plot/{i}"
    output_path=f"mvtec_results/weights/1.0/29/logs/plot/{i}.png"
    table(folder_path,output_path)
'''
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def table_with_highlight(folder_path, previous_paths, output_path, current_iteration):
    # Define paths for the current and previous iterations
    current_path = os.path.join(folder_path, str(current_iteration))
    previous_paths = [os.path.join(previous_paths, str(i)) for i in range(current_iteration)]

    # Gather lists of 'img' and 'ht' files for the current iteration
    img_files = sorted([f for f in os.listdir(current_path) if f.startswith('img')])
    ht_files = sorted([f for f in os.listdir(current_path) if f.startswith('ht')])

    # Load 'img' and 'ht' images, converting them to RGB
    img_list = [cv2.cvtColor(cv2.imread(os.path.join(current_path, f)), cv2.COLOR_BGR2RGB) for f in img_files]
    ht_list = [cv2.cvtColor(cv2.imread(os.path.join(current_path, f)), cv2.COLOR_BGR2RGB) for f in ht_files]

    # Identify labeled images from previous iterations
    labeled_images = set()
    for path in previous_paths:
        labeled_images.update(os.listdir(path))

    # Create a modified list with a red border added to labeled images
    bordered_img_list = []
    for i, img in enumerate(img_list):
        if img_files[i] in labeled_images:
            # Add a 5-pixel red border around labeled images
            red_border = [255, 0, 0]  # RGB color for red
            img_with_border = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=red_border)
            bordered_img_list.append(img_with_border)
        else:
            # No border for unlabeled images
            bordered_img_list.append(img)

    # Calculate the number of columns needed for the table
    n_cols = max(len(bordered_img_list), len(ht_list))

    # Set up a 2-row subplot: the first row for 'img' and the second row for 'ht'
    fig, axs = plt.subplots(2, n_cols, figsize=(2 * n_cols, 6))

    # Populate the subplot with images
    for col in range(n_cols):
        # Display 'img' images in the first row
        if col < len(bordered_img_list):
            axs[0, col].imshow(bordered_img_list[col])
            axs[0, col].axis('off')

        # Display 'ht' images in the second row
        if col < len(ht_list):
            axs[1, col].imshow(ht_list[col])
            axs[1, col].axis('off')

    # Save the final table as an image
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    '''
def table_with_highlight(folder_path, previous_paths, output_path, current_iteration):
    # Get the paths for the current and all previous iterations
    current_path = os.path.join(folder_path, str(current_iteration))
    previous_paths = [os.path.join(previous_paths, str(i)) for i in range(current_iteration)]

    # Get sorted lists of files for 'img' and 'ht' in the current iteration
    img_files = sorted([f for f in os.listdir(current_path) if f.startswith('img')])
    ht_files = sorted([f for f in os.listdir(current_path) if f.startswith('ht')])

    # Load images and convert to RGB
    img_list = [cv2.cvtColor(cv2.imread(os.path.join(current_path, f)), cv2.COLOR_BGR2RGB) for f in img_files]
    ht_list = [cv2.cvtColor(cv2.imread(os.path.join(current_path, f)), cv2.COLOR_BGR2RGB) for f in ht_files]

    # Identify labeled images from previous iterations
    labeled_images = set()
    for path in previous_paths:
        labeled_images.update(os.listdir(path))

    # Determine the maximum number of columns needed
    n_cols = max(len(img_list), len(ht_list))

    # Create a 2-row plot for img and ht images
    fig, axs = plt.subplots(2, n_cols, figsize=(2 * n_cols, 6))

    # Fill in each row of the plot
    for col in range(n_cols):
        if col < len(img_list):
            axs[0, col].imshow(img_list[col])
            axs[0, col].axis('off')

            # Add a red border if img file was labeled in previous iterations
            if img_files[col] in labeled_images:
                # Create a red rectangle around the image
                rect = patches.Rectangle((0, 0), img_list[col].shape[1], img_list[col].shape[0],
                                         linewidth=2, edgecolor='red', facecolor='none')
                axs[0, col].add_patch(rect)

        if col < len(ht_list):
            axs[1, col].imshow(ht_list[col])
            axs[1, col].axis('off')

    # Save the plot to the specified output path
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
'''
previous_path = "mvtec_results/weights/1.0/29/mask"
for i in range(1,10):
    folder_path=f"mvtec_results/weights/1.0/29/logs/plot"
    output_path=f"mvtec_results/weights/1.0/29/logs/plot/highlights/{i}.png"
    table_with_highlight(folder_path,previous_path,output_path,i)

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