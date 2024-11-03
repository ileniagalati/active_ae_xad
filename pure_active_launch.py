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

def select_pure_samples(X_train, Y_train, GT_train, scores, purity=0.5):
    '''
    Funzione che crea il nuovo dataset di addestramento in base alla frazione alpha=purity
    '''
    #indici di esempi unlabeled
    unlabeled_idx = np.where(Y_train == 0)[0]

    #indici selezionati (frazione alpha)
    selected_idx = np.argsort(scores[unlabeled_idx])[:int(purity * len(unlabeled_idx))]
    pure_indices = unlabeled_idx[selected_idx]

    #creiamo il nuovo dataset
    X_pure = X_train[pure_indices]
    X_pure = np.append(X_pure, X_train[Y_train == 1], axis=0)
    X_pure = np.append(X_pure, X_train[Y_train == -1], axis=0)

    Y_pure = Y_train[pure_indices]
    Y_pure = np.append(Y_pure, Y_train[Y_train == 1], axis=0)
    Y_pure = np.append(Y_pure, Y_train[Y_train == -1], axis=0)

    GT_pure = GT_train[pure_indices]
    GT_pure = np.append(GT_pure, GT_train[Y_train == 1], axis=0)
    GT_pure = np.append(GT_pure, GT_train[Y_train == -1], axis=0)

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

    X_test = X_train
    Y_test = Y_train
    GT_test = GT_train

    return X_train, Y_train, GT_train, X_test, Y_test, GT_test, lambda_u, lambda_n, lambda_a

if __name__ == '__main__':

    print("is cuda available: ",torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-budget', type=int, help='Budget')
    parser.add_argument('-epochs', type=int, help='Epochs to train')
    parser.add_argument('-seed', type=int, help='Seed')
    parser.add_argument('-purity', type=float, default=0.5, help='Purity parameter for active learning')
    args = parser.parse_args()

    b = args.budget
    s = args.seed
    purity = args.purity

    dataset_path= f'datasets/{args.ds}'

    if args.ds == 'brain':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test = \
            load_brainMRI_dataset(dataset_path)
    if args.ds == 'mvtec':
        X_train, Y_train, GT_train, X_test, Y_test, GT_test, GT_expert, Y_expert = \
            mvtec(5,dataset_path,10,seed=s)

    data_path = os.path.join('results','test_data', str(args.ds))
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

    #np.save(open(os.path.join(data_path, 'GT.npy'), 'wb'), GT_expert)
    ret_path = os.path.join('results','output', str(args.ds))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_expert)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_expert)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []

    lambda_u = n_examples / np.sum(Y_train == 0)
    lambda_n=0
    lambda_a=0

    print("first lambda_u: ", lambda_u)

    for x in range(0, b):
        heatmaps, scores, gtmaps, labels, tot_time = training_active_aexad(data_path,epochs=args.epochs,dataset=str(args.ds),
                                                                           lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a)

        active_images=os.path.join('results',"query",str(args.ds),str(x))
        if not os.path.exists(active_images):
            os.makedirs(active_images)


        np.save(open(os.path.join(ret_path, f'aexad_htmaps_{b}.npy'), 'wb'), heatmaps)
        np.save(open(os.path.join(ret_path, f'aexad_scores_{b}.npy'), 'wb'), scores)

        idx = np.argsort(scores[Y_train == 0])[::-1]

        img="a"
        ext=".png"

        #query selection
        query = X_train[Y_train == 0][idx[0]]
        img_to_save = Image.fromarray(query.astype(np.uint8))
        img_to_save.save(os.path.join(active_images, img+ext))
        print("dim image: ", query.shape)

        mask_images=os.path.join('results',"mask",str(args.ds),str(x))
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

        #aggiornamento del dataset
        X_train, Y_train, GT_train, X_test, Y_test, GT_test, lambda_u, lambda_n, lambda_a = \
            update_datasets(idx[0], mask_array, X_train, Y_train, GT_train)

        #seleziono la frazione alpha di esempi per il training che minimizzano l'anomaly score
        X_pure, Y_pure, GT_pure, pure_indices = select_pure_samples(X_train, Y_train, GT_train,scores,purity)

        print("pure indices: ", pure_indices)
        print("new dataset len: ", len(pure_indices))
        print("new X_pure shape: ", X_pure.shape)
        print("new Y_pure shape: ", Y_pure.shape)
        print("new GT_pure shape: ", GT_pure.shape)

        np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_pure)
        np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_pure)
        np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_pure)

        np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)


    heatmaps, scores, _, _, tot_time = training_active_aexad(data_path,epochs=args.epochs,dataset=str(args.ds),
                                                             lambda_u = lambda_u, lambda_n = lambda_n, lambda_a = lambda_a)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_f.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_f.npy'), 'wb'), scores)
