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
    parser.add_argument('-ie', type=int, help="epoche per le iterazioni dopo la prima")
    parser.add_argument('-s', type=int, help='Seed')
    parser.add_argument('-p', type=float, default=0.5, help='Purity parameter for active learning')
    parser.add_argument('-l', type=float, default=1, help='0: starting training from last iteration; 1: starting training from scratch')
    parser.add_argument('-r', type=str, default='results', help='Results path')
    args = parser.parse_args()

    b = args.b
    epochs= args.e
    s = args.s
    purity = args.p
    l=bool(args.l)
    root=args.r
    it_epochs = args.ie

    dataset_path = args.ds
    ds = os.path.basename(dataset_path)


    X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT_expert, Y_expert = \
            mvtec(6,dataset_path,10,seed=s)

    if l:
        c="scratch"
    else:
        c="weights"
    ret_path = os.path.join(root,c, str(purity), str(s))

    data=os.path.join(ret_path,'test_data')
    if not os.path.exists(data):
        os.makedirs(data)

    data_path = os.path.join(data, str(0))
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

    latest_weights_path=os.path.join(data_path,'latest_model_weights_0.pt')

    for x in range(0, b+1):
        print("iterazione: ", x)
        print("su iterazioni: ", b)
        if(x == b+1):
            print("stop a ")

            print(x)
            print("su ", b)
            break
        if x > 0:
            epochs = it_epochs
            n_query = 1
            print("# di query: ", n_query)
        if x == 0:
            n_query = int(b / 2)
            print("# di query: ", n_query)
            b = int(b/2)

        if (x > 0 or x == 0) and not os.path.exists(os.path.join(data_path, f'latest_model_weights_{x}.pt')):
            print(f"training on {x} iteration")
            if(x!=0):
                heatmaps, scores, _,_, tot_time, output = training_active_aexad(data_path=data_path,epochs=epochs,dataset=ds,
                                            lambda_p=None, lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a, save_path=data_path, times=times, l=l, iteration=x)

        data_path = os.path.join(data, str(x+1))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        active_images=os.path.join(ret_path,"query",str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        log_path = os.path.join(ret_path,'logs',str(x))

        if os.path.exists(log_path) and x == 0:
            heatmaps = np.load(open(os.path.join(log_path, f'aexad_htmaps_{x}.npy'), 'rb'))
            scores = np.load(open(os.path.join(log_path, f'aexad_scores_{x}.npy'), 'rb'))
            output = np.load(open(os.path.join(log_path, f'output_{x}.npy'), 'rb'))

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if x!=0:
            np.save(open(os.path.join(log_path, f'aexad_htmaps_{x}.npy'), 'wb'), heatmaps)
            np.save(open(os.path.join(log_path, f'aexad_scores_{x}.npy'), 'wb'), scores)
            np.save(open(os.path.join(log_path, f'output_{x}.npy'), 'wb'), output)

        #idx = [240, 250, 260, 270, 280, 289]
        idx = np.argsort(scores[Y_train == 0])[::-1]
        for ex in range (0,n_query):
            #if x>0:
                #idx = np.argsort(scores[Y_train == 0])[::-1]
            ext=".png"
            #img="a"
            print("ex: ", ex)
            img=f"{ex}"
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

            '''if x==0:
                print("idx pre delete: ", idx)
                idx = idx[1:]
                print("idx post delete: ",idx)'''

        #seleziono la frazione alpha di esempi per il training che minimizzano l'anomaly score
        #X_pure, Y_pure, GT_pure, pure_indices = select_pure_samples(X_train, Y_train, GT_train,scores,purity)

        #print(f"training on alpha = {purity} fraction of examples: ", len(pure_indices))

        X_pure=X_train
        Y_pure = Y_train
        GT_pure = GT_train

        np.save(open(os.path.join(data_path, f'X_train.npy'), 'wb'), X_pure)
        np.save(open(os.path.join(data_path, f'Y_train.npy'), 'wb'), Y_pure)
        np.save(open(os.path.join(data_path, f'GT_train.npy'), 'wb'), GT_pure)

        np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

        #update dei valori dei pesi della loss e normalizzazione degli stessi, sum=1
        n = len(X_pure)
        lambda_u = n / np.sum(Y_pure == 0) if np.sum(Y_pure == 0) > 0 else 0
        ''' lambda_n = min(50, n / np.sum(Y_pure == 1)) if np.sum(Y_pure == 1) > 0 else 0
        lambda_a = min(50, n / np.sum(Y_pure == -1)) if np.sum(Y_pure == -1) > 0 else 0'''

        lambda_n = n / np.sum(Y_pure == 1) if np.sum(Y_pure == 1) > 0 else 0
        lambda_a = n / np.sum(Y_pure == -1) if np.sum(Y_pure == -1) > 0 else 0

        print("unlabeled lambda: ", lambda_u)
        print("normal lambda: ", lambda_n)
        print("anomalous lambda: ", lambda_a)

    log_path = os.path.join(ret_path, 'logs', str(x))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    data_path = os.path.join(data, str(x))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    #training finale
    heatmaps, scores, _, _, tot_time, output = training_active_aexad(data_path,epochs=epochs,dataset=ds,
                                        lambda_p=None, lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a, save_path=data_path, times=times, l=l, iteration=x)

    np.save(open(os.path.join(log_path, f'aexad_htmaps_{x}.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(log_path, f'aexad_scores_{x}.npy'), 'wb'), scores)
    np.save(open(os.path.join(log_path, f'output_{x}.npy'), 'wb'), output)

    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
