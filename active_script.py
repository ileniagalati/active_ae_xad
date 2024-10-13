import argparse
import os
import shutil
import pickle
import matplotlib.pyplot as plt
import subprocess
import torch
from PIL import Image

from aexad.tools.create_dataset import load_brain_dataset
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

if __name__ == '__main__':
    dataset_path='brain_dataset'

    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-budget', type=int, help='Budget')
    args = parser.parse_args()
    b=args.budget

    if args.ds == 'brain':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            load_brain_dataset(dataset_path)

    data_path = os.path.join('test_data', str(args.ds), str(args.s))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    ret_path = os.path.join('results', str(args.ds), str(args.s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []
    path=os.path.join("results",str(args.ds), str(args.s))

    for x in range(0, b):

        heatmaps, scores, _, _, tot_time = training_active_aexad(data_path,epochs=1,dataset=str(args.ds))

        active_images=os.path.join("active_results",str(args.ds),str(args.s),str(b))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps.npy'), 'rb'))
        scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores.npy'), 'rb'))
        print(scores_aexad_conv)

        idx = np.argsort(scores_aexad_conv)[::-1]
        print(scores_aexad_conv[idx==0])
        print(scores_aexad_conv[idx==31])
        img="a"
        ext=".png"
        #query selection

        print(X_train.shape,'xtrain')
        print(Y_train.shape,'ytrain')
        print(X_test.shape,'xtest')
        print(Y_test.shape,'ytest')
        #print('image',X_train[idx[0]].shape)
        image_to_save = X_train[idx[0]]
        print("Dimensione dell'immagine da salvare:", image_to_save.shape)  # Dovrebbe stampare (256, 256, 3)

        # Salva l'immagine usando PIL
        img_to_save = Image.fromarray(image_to_save.astype(np.uint8))  # Assicurati che sia uint8
        img_to_save.save(os.path.join(active_images, 'image_name.png'))  # Modifica img e ext come necessario

        print("Immagine salvata con successo!")
        '''
        plot_image_tosave(X_train[idx[0]])
        plt.savefig(os.path.join(active_images,img+ext))
        '''
        mask_images=os.path.join("mask_results",str(args.ds),str(args.s),str(b))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)

        run_mask_generation(from_path,to_path)
        print("finished iteration")

        #image = np.array(Image.open(from_path).convert('RGB').resize(28,28))
        #gt_image= np.array(Image.open(to_path).convert('RGB').resize(28,28))
        #X_train.append(image)
        #GT_train.append(np.zeros_like(image, dtype=np.uint8))

    shutil.rmtree(data_path)