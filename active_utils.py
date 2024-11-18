import os
from active_aexad_script import launch as launch_aexad
import numpy as np
import subprocess
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances

def f(x):
    return 1 - x

def select_pure_samples(X_train, Y_train, GT_train, scores, purity=0.5):
    #indici di esempi unlabeled
    unlabeled_idx = np.where(Y_train == 0)[0]
    print("unlabled_idx", len(unlabeled_idx))

    #indici selezionati (frazione alpha)
    selected_idx = np.argsort(scores[unlabeled_idx])[:int(purity * len(unlabeled_idx))]
    pure_indices = unlabeled_idx[selected_idx]

    #creazione del nuovo dataset
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

def training_active_aexad(data_path, epochs, dataset, lambda_p, lambda_u, lambda_n, lambda_a, save_path, times, l,iteration=0):
    heatmaps, scores, _, _, tot_time, output = launch_aexad(data_path, epochs,8, 32,lambda_p, lambda_u, lambda_n, lambda_a, f=f, AE_type='conv',
                                                    save_intermediate=True, save_path=save_path, dataset=dataset, loss='aaexad', restart_from_scratch=l, iteration=iteration)
    times.append(tot_time)
    np.save(open(os.path.join(save_path, 'times.npy'), 'wb'), np.array(times))

    return heatmaps, scores, _, _, tot_time, output

def update_datasets(image_idx, mask_array, X_train, Y_train, GT_train):
    indices_zero = np.where(Y_train == 0)[0]

    if np.sum(mask_array) > 0:
        Y_train[indices_zero[image_idx]] = -1

        GT_train[indices_zero[image_idx]] = mask_array
    else:
        Y_train[indices_zero[image_idx]] = 1
        GT_train[indices_zero[image_idx]] = mask_array

    X_test = X_train
    Y_test = Y_train
    GT_test = GT_train

    return X_train, Y_train, GT_train, X_test, Y_test, GT_test

def run_mask_generation(from_path,to_path):
    subprocess.run(["python3", "MaskGenerator.py" ,"-from_path",from_path,"-to_path",to_path])


def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def Kmeans_dist(embs, K, tau=0.1):
    idx_active = []
    dist_matrix = torch.cdist(embs, embs, p=2).cpu().numpy()
    dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())
    dist_matrix = dist_matrix.astype(np.float64)
    dist_matrix = np.exp(dist_matrix / tau)
    idx_ = np.argmin(np.mean(dist_matrix, 0))

    idx_active.append(idx_)

    while len(idx_active) < K:
        p = dist_matrix[idx_active].min(0)
        p = p / p.sum()

        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(p)), p))
        idx_ = customDist.rvs(size=1)[0]
        while idx_ in idx_active:
            idx_ = customDist.rvs(size=1)[0]
        idx_active.append(idx_)

    return idx_active
