import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from aexad.tools.evaluation_metrics import Xauc
ds="mvtec"
htmap_stats = []
det_stats = []

ret_path = "mvtec_results/leather/kmeans/500ep_75iep_dlr01_to_0.0005_10b_minmax/weights/1.0/29"
GT = np.load(open(os.path.join(ret_path,"output", 'gt.npy'), 'rb'))
Y = np.load(open(os.path.join(ret_path,"output", 'labels.npy'), 'rb'))
# sistemare la size delle gt
gt_resized = np.zeros((GT.shape[0], 3, 256, 256), dtype=np.uint8)
for i in range(GT.shape[0]):
    img = Image.fromarray(GT[i])
    img_resized = img.resize((256, 256),Image.Resampling.NEAREST)
    gt_resized[i] = np.transpose(np.array(img_resized), (2, 0, 1))
GT=gt_resized

try:
    for i in range (1,1000):

        htmaps_aexad = np.load(open(os.path.join(ret_path,"logs", str(i), f'aexad_htmaps_{i}.npy'), 'rb'))
        scores_aexad = np.load(open(os.path.join(ret_path,"logs", str(i), f'aexad_scores_{i}.npy'), 'rb'))

        # explanation e detection
        htmap_stats.append(Xauc(GT[Y == 1], htmaps_aexad[Y == 1]))
        det_stats.append(roc_auc_score(Y, scores_aexad))
except:
    print("GT: ", GT.shape)
    print("Y: ", Y.shape)
    print("ht: ", htmaps_aexad.shape)
    print("scores: ", scores_aexad.shape)

    htmap_stats = np.array(htmap_stats)
    det_stats = np.array(det_stats)
    print("Explanation")
    print(htmap_stats)
    print("Detection")
    print(det_stats)

    with open(os.path.join(ret_path,'output.txt'), 'w') as f:
        f.write("Explanation:\n")
        np.savetxt(f, htmap_stats, fmt='%s')  # Salva htmap_stats
        f.write("\nDetection:\n")
        np.savetxt(f, det_stats, fmt='%s')  # Salva det_stats

    print("I dati sono stati salvati nel file output.txt")