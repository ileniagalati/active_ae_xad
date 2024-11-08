import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from aexad.tools.evaluation_metrics import Xauc
ds="mvtec"
htmap_stats = []
det_stats = []

ret_path = "mvtec_results/decreasing queries/weights/1.0/29"
GT = np.load(open(os.path.join(ret_path,"output", 'gt.npy'), 'rb'))
Y = np.load(open(os.path.join(ret_path,"output", 'labels.npy'), 'rb'))
#Y_test = np.load(open(os.path.join(ret_path,'test_data', 'Y_test.npy'), 'rb'))

for i in range (0,5):

    if i==5:
        i='f'

    htmaps_aexad = np.load(open(os.path.join(ret_path,"output", f'aexad_htmaps_{i}.npy'), 'rb'))
    scores_aexad = np.load(open(os.path.join(ret_path,"output", f'aexad_scores_{i}.npy'), 'rb'))

    # explanation e detection
    htmap_stats.append(Xauc(GT[Y == 1], htmaps_aexad[Y == 1]))
    det_stats.append(roc_auc_score(Y, scores_aexad))

# sistemare la size delle gt
gt_resized = np.zeros((GT.shape[0], 3, 256, 256), dtype=np.uint8)
for i in range(GT.shape[0]):
    img = Image.fromarray(GT[i])
    img_resized = img.resize((256, 256),Image.NEAREST)
    gt_resized[i] = np.transpose(np.array(img_resized), (2, 0, 1))
GT=gt_resized

htmap_stats = np.array(htmap_stats)
det_stats = np.array(det_stats)
print("Explanation")
print(htmap_stats)
print("Detection")
print(det_stats)