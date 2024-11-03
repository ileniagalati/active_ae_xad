import os
from active_aexad_script import launch as launch_aexad
import numpy as np

def f(x):
    return 1 - x

def select_pure_samples(X_train, Y_train, GT_train, scores, purity=0.5):
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
def training_active_aexad(data_path, epochs, dataset, lambda_u, lambda_n, lambda_a, ret_path, times, l):
    heatmaps, scores, _, _, tot_time = launch_aexad(data_path, epochs, 16, 32, lambda_u, lambda_n, lambda_a, f=f, AE_type='conv',
                                                    save_intermediate=True, save_path=ret_path, dataset=dataset, loss='aaexad',restart_from_scratch=l)
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

    X_test = X_train
    Y_test = Y_train
    GT_test = GT_train

    return X_train, Y_train, GT_train, X_test, Y_test, GT_test