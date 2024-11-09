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




if __name__ == '__main__':

    print("is cuda available: ",torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-b', type=int, help='Budget')
    parser.add_argument('-e', type=int, help='Epochs to train')
    parser.add_argument('-s', type=int, help='Seed')
    parser.add_argument('-p', type=float, default=0.5, help='Purity parameter for active learning')
    parser.add_argument('-l', type=float, default=1, help='0: starting training from last iteration; 1: starting training from scratch')
    parser.add_argument('-r', type=str, default='results', help='Results path')
    parser.add_argument('--debug', action='store_true', help="Attiva il debug mode")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    b = args.b
    epochs= args.e
    s = args.s
    purity = args.p
    l=bool(args.l)
    root=args.r

    dataset_path = args.ds
    ds = os.path.basename(dataset_path)


    X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT_expert, Y_expert = \
            mvtec(5,dataset_path,10,seed=s)

    if l:
        c="scratch"
    else:
        c="weights"
    ret_path = os.path.join(root,c, str(purity), str(s))

    data_path = os.path.join(ret_path,'test_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    n_examples=len(X_train)

    print("numero di esempi: ",n_examples)

    np.save(open(os.path.join(data_path, 'X_train_0.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'Y_train_0.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'GT_train_0.npy'), 'wb'), GT_train)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)

    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)


    o=os.path.join(ret_path,"output")
    if not os.path.exists(o):
        os.makedirs(o)
    np.save(open(os.path.join(o, 'gt.npy'), 'wb'), GT_expert)
    np.save(open(os.path.join(o, 'labels.npy'), 'wb'), Y_expert)

    pickle.dump(args, open(os.path.join(o, 'args'), 'wb'))

    times = []

    lambda_u = n_examples / np.sum(Y_train == 0)
    lambda_n=0
    lambda_a=0

    print("first lambda_u: ", lambda_u)

    for x in range(0, b):
        if x > 0:
            epochs = 10

        print(f"training on {x} iteration")
        heatmaps, scores, _,_, tot_time, output = training_active_aexad(data_path,epochs=epochs,dataset=ds,
                                        lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a, ret_path=ret_path, times=times, l=l)

        active_images=os.path.join(ret_path,"query",str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        log_path = os.path.join(ret_path,'logs',str(x))
        if not os.path.exists(log_path):
            os.makedirs(log_path)


        np.save(open(os.path.join(log_path, f'aexad_htmaps_{x}.npy'), 'wb'), heatmaps)
        np.save(open(os.path.join(log_path, f'aexad_scores_{x}.npy'), 'wb'), scores)
        np.save(open(os.path.join(log_path, f'output_{x}.npy'), 'wb'), output)

        np.save(open(os.path.join(log_path, f'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(log_path, f'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(log_path, f'GT_test.npy'), 'wb'), GT_test)

        np.save(open(os.path.join(o, f'aexad_htmaps_{x}.npy'), 'wb'), heatmaps)
        np.save(open(os.path.join(o, f'aexad_scores_{x}.npy'), 'wb'), scores)

        idx = np.argsort(scores[Y_train == 0])[::-1]


        ext=".png"
        img="a"
        #query selection
        query = X_train[Y_train == 0][idx[0]]
        img_to_save = Image.fromarray(query.astype(np.uint8))
        img_to_save.save(os.path.join(active_images, img+ext))
        #print("dim image: ", query.shape)

        mask_images=os.path.join(ret_path,"mask",str(x))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)

        #generazione della maschera manuale
        #run_mask_generation(from_path,to_path)

        #generazione della maschera dal dataset etichettato
        mask_from_gt = GT_expert[Y_train == 0][idx[0]]
        mask_img = Image.fromarray(mask_from_gt.astype(np.uint8))
        mask_img.save(to_path)

        #aggiornamento del dataset
        mask_img = Image.open(to_path)
        mask_array = np.array(mask_img)

        X_train, Y_train, GT_train, X_test, Y_test, GT_test = \
            update_datasets(idx[0], mask_array, X_train, Y_train, GT_train)

        #seleziono la frazione alpha di esempi per il training che minimizzano l'anomaly score
        X_pure, Y_pure, GT_pure, pure_indices = select_pure_samples(X_train, Y_train, GT_train,scores,purity)

        print(f"training on alpha = {purity} fraction of examples: ", len(pure_indices))

        np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_pure)
        np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_pure)
        np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_pure)

        np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

        #update dei valori dei pesi della loss e normalizzazione degli stessi, sum=1
        n = len(X_pure)
        lambda_u = n / np.sum(Y_pure == 0) if np.sum(Y_pure == 0) > 0 else 0
        lambda_n = n / np.sum(Y_pure == 1) if np.sum(Y_pure == 1) > 0 else 0
        lambda_a = n / np.sum(Y_pure == -1) if np.sum(Y_pure == -1) > 0 else 0

        print("unlabeled lambda: ", lambda_u)
        print("normal lambda: ", lambda_n)
        print("anomalous lambda: ", lambda_a)


    log_path = os.path.join(ret_path, 'logs', "f")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    #training finale
    heatmaps, scores, _, _, tot_time, output = training_active_aexad(data_path,epochs=epochs,dataset=ds,
                                        lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a, ret_path=ret_path, times=times, l=l)
    np.save(open(os.path.join(o, 'aexad_htmaps_f.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(o, 'aexad_scores_f.npy'), 'wb'), scores)

    np.save(open(os.path.join(log_path, 'aexad_htmaps_f.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(log_path, 'aexad_scores_f.npy'), 'wb'), scores)
    np.save(open(os.path.join(log_path, f'output_{x}.npy'), 'wb'), output)

    np.save(open(os.path.join(log_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(log_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(log_path, 'GT_test.npy'), 'wb'), GT_test)
