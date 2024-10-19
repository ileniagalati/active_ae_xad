import argparse
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import subprocess
import torch
from PIL import Image

from aexad.tools.create_dataset import load_brainMRI_dataset
from active_aexad_script import launch as launch_aexad
import numpy as np
from aexad.tools.utils import plot_image_tosave, plot_heatmap_tosave

def f(x):
    return 1-x

def training_active_aexad(data_path,epochs,dataset):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, (28*28) / 25, (28*28) / 25, f, 'conv',
                                                    save_intermediate=True, save_path=ret_path,dataset=dataset,loss='aaexad')
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)
    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))

    return heatmaps, scores, _, _, tot_time

def run_mask_generation(from_path,to_path):
    subprocess.run(["python3", "MaskGenerator.py" ,"-from_path",from_path,"-to_path",to_path])

def update_ground_truth_with_mask(image_idx, mask_path, GT_train, gt_save_path):
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img)
    GT_train[image_idx] = mask_array
    np.save(gt_save_path, GT_train)

def is_image_empty(image_path):
    mask_img = Image.open(image_path)
    mask_array = np.array(mask_img)
    return np.all(mask_array == 0)

def update_datasets(image_idx, mask_array, X_0, X_an, X_no, Y_an, Y_no, GT_an, GT_no):
    if np.sum(mask_array) > 0:
        X_an.append(X_0[image_idx])
        Y_an[image_idx] = 1
        GT_an[image_idx] = mask_array
    else:
        X_no.append(X_0[image_idx])
        Y_no[image_idx] = 0
        GT_no[image_idx] = mask_array

    X_0 = np.delete(X_0, image_idx, axis=0)

    return X_0, X_an, X_no, Y_an, Y_no, GT_an, GT_no


if __name__ == '__main__':
    dataset_path= 'datasets/brainMRI'

    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-budget', type=int, help='Budget')
    args = parser.parse_args()
    b=args.budget

    if args.ds == 'brain':
        X_0, X_an, X_no, Y_an, Y_no , GT_an, GT_no, X_test, Y_test, GT_test = \
            load_brainMRI_dataset(dataset_path)

    data_path = os.path.join('results','test_data', str(args.ds))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_0.npy'), 'wb'), X_0)
    np.save(open(os.path.join(data_path, 'X_an.npy'), 'wb'), X_an)
    np.save(open(os.path.join(data_path, 'X_no.npy'), 'wb'), X_no)
    np.save(open(os.path.join(data_path, 'Y_an.npy'), 'wb'), Y_an)
    np.save(open(os.path.join(data_path, 'Y_no.npy'), 'wb'), Y_no)
    np.save(open(os.path.join(data_path, 'GT_an.npy'), 'wb'), GT_an)
    np.save(open(os.path.join(data_path, 'GT_no.npy'), 'wb'), GT_no)

    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    ret_path = os.path.join('results','output', str(args.ds))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []

    for x in range(0, b):

        heatmaps, scores, _, _, tot_time = training_active_aexad(data_path,epochs=1,dataset=str(args.ds))

        active_images=os.path.join('results',"query",str(args.ds),str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        htmaps_aexad_conv = np.load(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'rb'))
        scores_aexad_conv = np.load(open(os.path.join(ret_path, 'aexad_scores.npy'), 'rb'))

        idx = np.argsort(scores_aexad_conv)[::-1]
        img="a"
        ext=".png"

        #query selection
        image_to_save = X_0[idx[0]]
        img_to_save = Image.fromarray(image_to_save.astype(np.uint8))
        img_to_save.save(os.path.join(active_images, img+ext))

        mask_images=os.path.join('results',"mask",str(args.ds),str(x))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)

        #generazione della maschera
        run_mask_generation(from_path,to_path)

        #aggiornamento del dataset
        mask_img = Image.open(to_path)
        mask_array = np.array(mask_img)

        # Print lengths before updating datasets
        print(f"Before update:")
        print(f"Length of X_0: {len(X_0)}")
        print(f"Length of X_an: {len(X_an)}")
        print(f"Length of X_no: {len(X_no)}")
        print(f"Length of Y_an: {len(Y_an)}")
        print(f"Length of Y_no: {len(Y_no)}")
        print(f"Length of GT_an: {len(GT_an)}")
        print(f"Length of GT_no: {len(GT_no)}")

        # Update datasets based on the mask
        X_0, X_an, X_no, Y_an, Y_no, GT_an, GT_no =\
            update_datasets( idx[0], mask_array, X_0, X_an, X_no, Y_an, Y_no, GT_an, GT_no)

        # Print lengths after updating datasets
        print(f"After update:")
        print(f"Length of X_0: {len(X_0)}")
        print(f"Length of X_an: {len(X_an)}")
        print(f"Length of X_no: {len(X_no)}")
        print(f"Length of Y_an: {len(Y_an)}")
        print(f"Length of Y_no: {len(Y_no)}")
        print(f"Length of GT_an: {len(GT_an)}")
        print(f"Length of GT_no: {len(GT_no)}")

        # Save the updated arrays
        np.save(open(os.path.join(data_path, 'X_0.npy'), 'wb'), X_0)
        np.save(open(os.path.join(data_path, 'X_an.npy'), 'wb'), np.array(X_an))
        np.save(open(os.path.join(data_path, 'X_no.npy'), 'wb'), np.array(X_no))
        np.save(open(os.path.join(data_path, 'Y_an.npy'), 'wb'), np.array(Y_an))
        np.save(open(os.path.join(data_path, 'Y_no.npy'), 'wb'), np.array(Y_no))
        np.save(open(os.path.join(data_path, 'GT_an.npy'), 'wb'), np.array(GT_an))
        np.save(open(os.path.join(data_path, 'GT_no.npy'), 'wb'), np.array(GT_no))

   # shutil.rmtree(data_path)