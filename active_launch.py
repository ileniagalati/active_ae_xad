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
    return 1-x

def training_active_aexad(data_path,epochs,dataset,lambda_u, lambda_n, lambda_a):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, lambda_u, lambda_n, lambda_a, f=f, AE_type='conv',
                                                    save_intermediate=True, save_path=ret_path,dataset=dataset,loss='aaexad')
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)
    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))

    return heatmaps, scores, _, _, tot_time

def run_mask_generation(from_path,to_path):
    subprocess.run(["python3", "MaskGenerator.py" ,"-from_path",from_path,"-to_path",to_path])

def update_datasets(image_idx, mask_array, X_train, Y_train, GT_train):

    if np.sum(mask_array) > 0:
        Y_train[image_idx] = -1
        GT_train[image_idx] = mask_array
    else:
        Y_train[image_idx] = 1
        GT_train[image_idx] = mask_array

    n = len(X_train)
    lambda_u = n / np.sum(Y_train == 0) if np.sum(Y_train == 0) > 0 else 0
    lambda_n = n / np.sum(Y_train == 1) if np.sum(Y_train == 1) > 0 else 0
    lambda_a = n / np.sum(Y_train == -1) if np.sum(Y_train == -1) > 0 else 0

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
    b=args.budget

    dataset_path= f'datasets/{args.ds}'

    if args.ds == 'brain':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test = \
            load_brainMRI_dataset(dataset_path)
    if args.ds == 'mvtec':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT = \
            mvtec(5,dataset_path,15)

    data_path = os.path.join('results','test_data', str(args.ds))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    n_examples=len(X_train)
    print("numero di esempi: ",n_examples)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)

    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

    np.save(open(os.path.join(data_path, 'GT.npy'), 'wb'), GT)
    ret_path = os.path.join('results','output', str(args.ds))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []

    lambda_u = n_examples / np.sum(Y_train == 0)
    lambda_n=0
    lambda_a=0

    print("first lambda_u: ", lambda_u)

    for x in range(0, b):

        heatmaps, scores, _, _, tot_time = training_active_aexad(data_path,epochs=15,dataset=str(args.ds),
                                                                 lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a)

        active_images=os.path.join('results',"query",str(args.ds),str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        htmaps_aexad_conv = np.load(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'rb'))
        scores_aexad_conv = np.load(open(os.path.join(ret_path, 'aexad_scores.npy'), 'rb'))

        idx = np.argsort(scores_aexad_conv[Y_train == 0])[::-1]
        img="a"
        ext=".png"

        #query selection
        image_to_save = X_train[Y_train==0][idx[0]]

        img_to_save = Image.fromarray(image_to_save.astype(np.uint8))
        img_to_save.save(os.path.join(active_images, img+ext))
        print("dim image: ", image_to_save.shape)

        mask_images=os.path.join('results',"mask",str(args.ds),str(x))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)

        #generazione della maschera manuale
        #run_mask_generation(from_path,to_path)

        #generazione della maschera dal dataset etichettato
        mask_from_gt = GT[Y_train == 0][idx[0]]
        mask_img = Image.fromarray(mask_from_gt.astype(np.uint8))
        to_path = os.path.join(mask_images, img + "_mask" + ext)
        mask_img.save(to_path)

        #aggiornamento del dataset
        mask_img = Image.open(to_path)
        mask_array = np.array(mask_img)
        print("dimensioni maschera: ", mask_array.shape)

        #aggiornamento del dataset
        X_train, Y_train, GT_train, lambda_u, lambda_n, lambda_a = \
            update_datasets(idx[0], mask_array, X_train, Y_train, GT_train)

        np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
        np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
        np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
