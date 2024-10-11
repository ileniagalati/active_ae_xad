import argparse
import os
import shutil
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk #install python-tk@3.10
from PIL import Image, ImageTk #install Pillow
import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import torch
from kornia.filters import gaussian_blur2d

from aexad.tools.create_dataset import load_brain_dataset
from aexad.aexad_script import launch as launch_aexad
import MaskGenerator
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk #install python-tk@3.10
from PIL import Image, ImageTk #install Pillow
import numpy as np
from PIL import ImageDraw
from aexad.tools.utils import plot_image_tosave, plot_heatmap_tosave

def f(x):
    return 1-x

def training_active_aexad(data_path,epochs,dataset):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, (28*28) / 25, None, f, 'conv',
                                                    save_intermediate=True, save_path=ret_path,dataset=dataset)
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

        idx = np.argsort(scores_aexad_conv[Y_test == 1])[::-1]
        img="a"
        ext=".png"
        #query selection

        print(X_train.size,'xtrain')
        print(Y_train.size,'ytrain')
        print(X_test.size,'xtest')
        print(Y_test.size,'ytest')

        plot_image_tosave(X_train[Y_train==1][idx[1]])
        plt.savefig(os.path.join(active_images,img+ext), bbox_inches='tight', pad_inches=0)

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