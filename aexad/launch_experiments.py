import argparse
import os
import shutil
import pickle

import numpy as np
import torch

from tools.create_dataset import square, square_diff
#from run_fcdd import launch as launch_fcdd
#from run_deviation import launch as launch_dev
from aexad_script import launch as launch_aexad

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', type=str, help='Dataset to use')
    parser.add_argument('-c', type=int, help='Considered class')
    parser.add_argument('-patr', type=float, default=2)
    parser.add_argument('-pate', type=float, default=10)
    parser.add_argument('-i', help='Modification intesity')
    parser.add_argument('-s', type=int, help='Seed to use')
    parser.add_argument('-size', type=int, help='Size of the square')
    args = parser.parse_args()

    if args.i != 'rand':
        args.i = float(args.i)

    if args.ds == 'mnist'  or args.ds == 'fmnist':
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=args.ds, seed=args.s)
    elif args.ds == 'mnist_diff':
        dataset = 'mnist'
        X_train, Y_train, X_test, Y_test, GT_train, GT_test = \
            square_diff(args.c, perc_anom_train=args.patr, perc_anom_test=args.pate, size=args.size,
                   intensity=args.i, DATASET=dataset, seed=args.s)


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

  #  htmaps, scores, _, _, tot_time = launch_dev(dataset_root=data_path, epochs=50)  # 50
  #  np.save(open(os.path.join(ret_path, 'deviation_htmaps.npy'), 'wb'), htmaps)
   # np.save(open(os.path.join(ret_path, 'deviation_scores.npy'), 'wb'), np.array(scores))
   # times.append(tot_time)
  # del htmaps, scores
  #  torch.cuda.empty_cache()

  #  np.save(open(os.path.join(ret_path, 'fcdd_htmaps.npy'), 'wb'), htmaps)
  #  np.save(open(os.path.join(ret_path, 'fcdd_scores.npy'), 'wb'), np.array(scores))
   # times.append(tot_time)
    #del htmaps, scores
    #torch.cuda.empty_cache()

    def f(x):
        return 1-x

    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1, 16, 32, (28*28) / 25, None, f, 'shallow',
                                                    save_intermediate=True, save_path=ret_path)
    np.save(open(os.path.join(ret_path, 'aexad_htmaps.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, 'aexad_scores.npy'), 'wb'), scores)

    #heatmaps, scores, _, _, tot_time = launch_aexad(data_path, 1000, 16, 32, (28*28) / 25, None, f, 'conv',
    #                                                save_intermediate=True, save_path=ret_path)
    #np.save(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'wb'), heatmaps)
    #np.save(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'wb'), scores)

    times.append(tot_time)
    np.save(open(os.path.join(ret_path, 'times.npy'), 'wb'), np.array(times))
    print(times)

    shutil.rmtree(data_path)


