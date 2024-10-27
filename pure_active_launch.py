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

def f(x):
    return 1 - x

# Function to select samples based on purity parameter
def select_pure_samples(X_train, Y_train, GT_train, scores, purity=0.5):
    # Get indices of unlabeled examples
    unlabeled_idx = np.where(Y_train == 0)[0]

    # Sort scores of unlabeled examples in ascending order and select the most "pure" half
    selected_idx = np.argsort(scores[unlabeled_idx])[:int(purity * len(unlabeled_idx))]
    pure_indices = unlabeled_idx[selected_idx]

    # Create purified subsets
    X_pure = X_train[pure_indices]
    Y_pure = Y_train[pure_indices]
    GT_pure = GT_train[pure_indices]

    return X_pure, Y_pure, GT_pure, pure_indices

# Main training function
def training_active_aexad(data_path, epochs, dataset, lambda_u, lambda_n, lambda_a):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, lambda_u, lambda_n, lambda_a, f=f, AE_type='conv',
                                                    save_intermediate=True, save_path=ret_path, dataset=dataset, loss='aaexad')
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)
    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))

    return heatmaps, scores, _, _, tot_time

# Dataset update function
def update_datasets(image_idx, mask_array, X_train, Y_train, GT_train):
    if np.sum(mask_array) > 0:
        Y_train[image_idx] = -1
        GT_train[image_idx] = mask_array
    else:
        Y_train[image_idx] = 1
        GT_train[image_idx] = mask_array

    print("unlabeled examples: ", np.sum(Y_train == 0))
    print("normal examples: ", np.sum(Y_train == 1))
    print("anomalous examples: ", np.sum(Y_train == -1))

    lambda_u = 1 / np.sum(Y_train == 0) if np.sum(Y_train == 0) > 0 else 0
    lambda_n = 1 / np.sum(Y_train == 1) if np.sum(Y_train == 1) > 0 else 0
    lambda_a = 1 / np.sum(Y_train == -1) if np.sum(Y_train == -1) > 0 else 0

    print("unlabeled lambda: ", lambda_u)
    print("normal lambda: ", lambda_n)
    print("anomalous lambda: ", lambda_a)

    return X_train, Y_train, GT_train, lambda_u, lambda_n, lambda_a

if __name__ == '__main__':
    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-budget', type=int, help='Budget')
    args = parser.parse_args()
    b = args.budget

    dataset_path = f'datasets/{args.ds}'

    if args.ds == 'brain':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test = load_brainMRI_dataset(dataset_path)
    elif args.ds == 'mvtec':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT = mvtec(5, dataset_path, 15)

    data_path = os.path.join('results', 'test_data', str(args.ds))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)

    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

    np.save(open(os.path.join(data_path, 'GT.npy'), 'wb'), GT)
    ret_path = os.path.join('results', 'output', str(args.ds))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)

    times = []

    lambda_u = 1 / np.sum(Y_train == 0)
    lambda_n = 0
    lambda_a = 0
    print("Initial lambda_u:", lambda_u)

    for x in range(0, b):
        heatmaps, scores, _, _, tot_time = training_active_aexad(data_path, epochs=1, dataset=str(args.ds),
                                                                 lambda_u=lambda_u, lambda_n=lambda_n, lambda_a=lambda_a)

        # Select "pure" subset of unlabeled data for training
        X_pure, Y_pure, GT_pure, selected_indices = select_pure_samples(X_train, Y_train, GT_train, scores, purity=0.5)

        active_images = os.path.join('results', "query", str(args.ds), str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        # Save the most "pure" image for manual or labeled mask generation
        image_to_save = X_pure[0]
        img_to_save = Image.fromarray(image_to_save.astype(np.uint8))
        img_to_save.save(os.path.join(active_images, 'a.png'))

        mask_images = os.path.join('results', "mask", str(args.ds), str(x))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path = os.path.join(active_images, 'a.png')
        to_path = os.path.join(mask_images, 'a_mask.png')

        # Generating mask from labeled dataset
        mask_from_gt = GT_pure[0]
        mask_img = Image.fromarray(mask_from_gt.astype(np.uint8))
        mask_img.save(to_path)

        # Update the dataset with the selected "pure" sample
        mask_array = np.array(mask_img)
        X_train, Y_train, GT_train, lambda_u, lambda_n, lambda_a = update_datasets(selected_indices[0], mask_array, X_train, Y_train, GT_train)

        # Save updated data
        np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
        np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
        np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
