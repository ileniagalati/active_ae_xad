import argparse
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import subprocess
import torch
from PIL import Image

from aexad.tools.create_dataset import load_brainMRI_dataset
from aexad.aexad_script import launch as launch_aexad
import numpy as np
from aexad.tools.utils import plot_image_tosave, plot_heatmap_tosave

def f(x):
    return 1-x

def training_active_aexad(data_path,epochs,dataset):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, (256*256) / 25, None, f, 'conv',
                                                    save_intermediate=True, save_path=ret_path,dataset=dataset,loss='mse')
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

if __name__ == '__main__':
    dataset_path= 'datasets/brainMRI'

    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-budget', type=int, help='Budget')
    args = parser.parse_args()
    b=args.budget

    if args.ds == 'brain':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            load_brainMRI_dataset(dataset_path)

    data_path = os.path.join('results','test_data', str(args.ds))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
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
        image_to_save = X_train[idx[0]]
        img_to_save = Image.fromarray(image_to_save.astype(np.uint8))  # Assicurati che sia uint8
        img_to_save.save(os.path.join(active_images, img+ext))  # Modifica img e ext come necessario

        mask_images=os.path.join('results',"mask",str(args.ds),str(x))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)

        #generazione della maschera
        run_mask_generation(from_path,to_path)

        #aggiornamento della groud truth con la nuova maschera
        image_idx = idx[0]
        mask_path = to_path
        gt_save_path = os.path.join(data_path, 'GT_train.npy')

        GT_train = np.load(open(gt_save_path, 'rb'))
        update_ground_truth_with_mask(image_idx, mask_path, GT_train, gt_save_path)

   # shutil.rmtree(data_path)