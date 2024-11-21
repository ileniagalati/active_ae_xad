import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from aexad.tools.evaluation_metrics import Xauc

#CAMBIA QUI---------------------
ds="mnist"

#plot results
htmap_stats = []
det_stats = []
patr = [0.01,0.02, 0.03, 0.04]
for p in patr:
  if p==0.02:
    for i in range(10):
      ret2perc = os.path.join('/content/drive/MyDrive/AE_XAD/results/', ds, str(i),"29")
      GT_test = np.load(open(os.path.join(ret2perc, 'gt.npy'), 'rb'))
      Y_test = np.load(open(os.path.join(ret2perc, 'labels.npy'), 'rb'))
      htmaps_aexad = np.load(open(os.path.join(ret2perc, 'aexad_htmaps_conv.npy'), 'rb'))
      scores_aexad = np.load(open(os.path.join(ret2perc, 'aexad_scores_conv.npy'), 'rb'))
      htmap_stats.append(Xauc(GT_test[Y_test == 1], htmaps_aexad[Y_test == 1]))
      det_stats.append(roc_auc_score(Y_test, scores_aexad))
  else:
    for i in range(10):
        ret_path = os.path.join('results', 'patr', f'{p}_5', ds, str(i), "19")
        GT_test = np.load(open(os.path.join(ret_path, 'gt.npy'), 'rb'))
        Y_test = np.load(open(os.path.join(ret_path, 'labels.npy'), 'rb'))
        htmaps_aexad = np.load(open(os.path.join(ret_path, 'aexad_htmaps_conv.npy'), 'rb'))
        scores_aexad = np.load(open(os.path.join(ret_path, 'aexad_scores_conv.npy'), 'rb'))
        htmap_stats.append(Xauc(GT_test[Y_test == 1], htmaps_aexad[Y_test == 1]))
        det_stats.append(roc_auc_score(Y_test, scores_aexad))


htmap_stats = np.array(htmap_stats).reshape((len(patr), 10))
det_stats = np.array(det_stats).reshape((len(patr), 10))

plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.title('Detection')
for i in range(10):
    plt.plot(det_stats[:, i])
    plt.ylim(0.0, 1.1)
    plt.xticks(range(len(patr)), patr)

plt.subplot(2, 3, 2)
plt.title('Explanation')
for i in range(10):
    plt.plot(htmap_stats[:, i])
    plt.ylim(0.75, 1.1)
    plt.xticks(range(len(patr)), patr)

plt.subplot(2, 3, 3)
plt.title('Detection & Explanation')
for i in range(10):
    plt.plot(det_stats[:, i], label=f'Detection {i}')
    plt.plot(htmap_stats[:, i], label=f'Explanation {i}')
    plt.legend()
    plt.ylim(0.5, 1.001)
    plt.xticks(range(len(patr)), patr)

print(htmap_stats)
print(det_stats)

plt.tight_layout()
plt.show()