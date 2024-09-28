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

from aexad.tools.create_dataset import square
from aexad.aexad_script import launch as launch_aexad
import MaskGenerator
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk #install python-tk@3.10
from PIL import Image, ImageTk #install Pillow
import numpy as np
from PIL import ImageDraw
from aexad.tools.utils import plot_image_tosave

def f(x):
    return 1-x

def training_active_aexad(data_path,epochs):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, (28*28) / 25, None, f, 'conv',
                                                    save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)
    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))

    return heatmaps, scores, _, _, tot_time

def run_mask_generation(from_path,to_path):
    subprocess.run(["python3", "MaskGenerator.py" ,"-from_path",from_path,"-to_path",to_path])

if __name__ == '__main__':

    print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-i', help='Modification intesity')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    parser.add_argument('-budget', type=int, help='Budget')
    args = parser.parse_args()
    b=args.budget

    if args.i != 'rand':
        args.i = float(args.i)

    if args.ds == 'mnist':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=args.ds, seed=args.s)

    data_path = os.path.join('test_data', args.ds, str(args.c), str(args.s))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    ret_path = os.path.join('results', args.ds, str(args.c), str(args.s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args'), 'wb'))

    times = []
    path=os.path.join("results/mnist", str(args.c), '29')

    for x in range(0, b):

        heatmaps, scores, _, _, tot_time = training_active_aexad(data_path,epochs=1)

        active_images=os.path.join("active_results",str(args.ds),"class_"+str(args.c))
        if not os.path.exists(active_images):
            os.makedirs(active_images)

        htmaps_aexad_conv = np.load(open(os.path.join(path, 'aexad_htmaps.npy'), 'rb'))
        scores_aexad_conv = np.load(open(os.path.join(path, 'aexad_scores.npy'), 'rb'))

        idx = np.argsort(scores_aexad_conv[Y_test == 1])[::-1]
        htmaps_f = gaussian_blur2d(torch.from_numpy(htmaps_aexad_conv), kernel_size=(7, 7),sigma=(2, 2))

        img="a"
        ext=".png"

        #query selection per il momento prendo l'immagine con il deviation score più alto
        plot_image_tosave(X_train[Y_train==1][idx[1]])
        plt.savefig(os.path.join(active_images,img+ext))

        mask_images=os.path.join("mask_results",str(args.ds),"class_"+str(args.c))
        if not os.path.exists(mask_images):
            os.makedirs(mask_images)

        from_path=os.path.join(active_images,img+ext)
        to_path=os.path.join(mask_images,img+"_mask"+ext)
        #if not os.path.exists(to_path):
         #   os.makedirs(to_path)
        run_mask_generation(from_path,to_path)
        print("finished iteration")

    shutil.rmtree(data_path)