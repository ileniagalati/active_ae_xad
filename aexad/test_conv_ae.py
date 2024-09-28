import argparse
import os
import shutil
import pickle

import numpy as np
import torch

from tools.create_dataset import square, square_diff, mvtec
from aexad_script import launch as launch_aexad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-na', type=int, default=1, help='Number of anomalies for anomaly class (only for real ds)')
    parser.add_argument('-i', help='Modification intesity')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    parser.add_argument('-f', type=int, help='Function to use for mapping')
    parser.add_argument('-net', type=str, choices=['shallow', 'deep', 'conv', 'conv_deep', 'conv_f2'], help='Network to use')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    args = parser.parse_args()

    if args.i != 'rand' and (args.ds == 'mnist' or args.ds == 'mnist_diff'):
        args.i = float(args.i)

    if args.ds == 'mnist' or args.ds == 'fmnist':
        dataset = args.ds
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=args.ds, seed=args.s)
    elif args.ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=dataset, seed=args.s)
    elif args.ds == 'mvtec':
        dataset = args.ds
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = mvtec(args.c, 'datasets/mvtec', args.na, seed=args.s)

    data_path = os.path.join('test_data', args.ds, str(args.c), str(args.s))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
    np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
    np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
    np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)
    np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)
    # f_o funzione vecchia
    ret_path = os.path.join('results', f'f_{args.f}', args.ds, str(args.c), str(args.s))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    np.save(open(os.path.join(ret_path, 'gt.npy'), 'wb'), GT_test)
    np.save(open(os.path.join(ret_path, 'labels.npy'), 'wb'), Y_test)

    pickle.dump(args, open(os.path.join(ret_path, 'args_aexad'), 'wb'))

    if args.f == 0:
        # f_0
        def f(x):
           return 1 - x

    elif args.f == 1:
        #f_1
        def f(x):
            return torch.where(x >= 0.5, 0.0, 1.0)

    elif args.f == 2:
        def f(x):
            return x + 2


    heatmaps, scores, _, _, _ = launch_aexad(data_path, 2000, 16, 64, None, None, f, args.net,
                                                    save_intermediate=True, save_path=ret_path, use_cuda=args.cuda,
                                             dataset=dataset)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'wb'), scores)

    shutil.rmtree(data_path)


