import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from aexad.tools.evaluation_metrics import Xauc

# Percorsi per i diversi dataset
ds = "mvtec"
#b
ret_path_hazelnut = "mvtec_results/hazelnut/b/500ep_decay_lr_20ep_iteration/weights/1.0/29"
ret_path_leather = "mvtec_results/leather/b/500ep_decay_lr_20ep_iteration/weights/1.0/29"
ret_path_bottle = "mvtec_results/bottle/b/500ep_decay_lr_20ep_iteration/weights/1.0/29"
#b_split
'''ret_path_hazelnut = "mvtec_results/hazelnut/b_split/500ep_50iep_dlr01_to0.0005_b10/weights/1.0/29"
ret_path_leather = "mvtec_results/leather/b/500ep_decay_lr_20ep_iteration/weights/1.0/29"
ret_path_bottle = "mvtec_results/bottle/b/500ep_decay_lr_20ep_iteration/weights/1.0/29"'''


out = "mvtec_results/auc_plot/b"

x_values = [0,1,2,3,4,5,6,7,8,9]

# Variabili per le metriche di valutazione
htmap_stats_bottle = []
det_stats_bottle = []

htmap_stats_hazelnut = []
det_stats_hazelnut = []

htmap_stats_leather = []
det_stats_leather = []

def process_data(ret_path, htmap_stats, det_stats, dataset_name):
    print(f"Elaborazione del dataset: {dataset_name}")

    if dataset_name == "Bottle":

        GT = np.load(open(os.path.join(ret_path_bottle, "output", 'gt.npy'), 'rb'))
        Y = np.load(open(os.path.join(ret_path_bottle, "output", 'labels.npy'), 'rb'))

        gt_resized = np.zeros((GT.shape[0], 3, 256, 256), dtype=np.uint8)
        for i in range(GT.shape[0]):
            img = Image.fromarray(GT[i])
            img_resized = img.resize((256, 256), Image.NEAREST)
            gt_resized[i] = np.transpose(np.array(img_resized), (2, 0, 1))
        GT = gt_resized

    if dataset_name == "Hazelnut":

        GT = np.load(open(os.path.join(ret_path_hazelnut, "output", 'gt.npy'), 'rb'))
        Y = np.load(open(os.path.join(ret_path_hazelnut, "output", 'labels.npy'), 'rb'))

        gt_resized = np.zeros((GT.shape[0], 3, 256, 256), dtype=np.uint8)
        for i in range(GT.shape[0]):
            img = Image.fromarray(GT[i])
            img_resized = img.resize((256, 256), Image.NEAREST)
            gt_resized[i] = np.transpose(np.array(img_resized), (2, 0, 1))
        GT = gt_resized

    if dataset_name == "Leather":

        GT = np.load(open(os.path.join(ret_path_leather, "output", 'gt.npy'), 'rb'))
        Y = np.load(open(os.path.join(ret_path_leather, "output", 'labels.npy'), 'rb'))

        gt_resized = np.zeros((GT.shape[0], 3, 256, 256), dtype=np.uint8)
        for i in range(GT.shape[0]):
            img = Image.fromarray(GT[i])
            img_resized = img.resize((256, 256), Image.NEAREST)
            gt_resized[i] = np.transpose(np.array(img_resized), (2, 0, 1))
        GT = gt_resized

    for i in range(11):
        try:
            Y_test = np.load(open(os.path.join(ret_path, 'test_data', str(i), 'Y_test.npy'), 'rb'))
            htmaps_aexad = np.load(open(os.path.join(ret_path, "logs", str(i), f'aexad_htmaps_{i}.npy'), 'rb'))
            scores_aexad = np.load(open(os.path.join(ret_path, "logs", str(i), f'aexad_scores_{i}.npy'), 'rb'))

            scores_Y_test_0 = scores_aexad[Y_test == 0]
            max_score = np.max(scores_Y_test_0)
            min_score = np.min(scores_Y_test_0)

            adjusted_scores = scores_aexad.copy()
            adjusted_scores[Y_test == -1] = max_score
            adjusted_scores[Y_test == 1] = min_score

            htmap_stats.append(Xauc(GT[Y == 1], htmaps_aexad[Y == 1]))
            det_stats.append(roc_auc_score(Y, adjusted_scores))
        except Exception as e:
            print(f"Errore nel processo per l'iterazione {i} nel dataset {dataset_name}: {e}")
            break

process_data(ret_path_bottle, htmap_stats_bottle, det_stats_bottle, 'Bottle')
process_data(ret_path_hazelnut, htmap_stats_hazelnut, det_stats_hazelnut, 'Hazelnut')
process_data(ret_path_leather, htmap_stats_leather, det_stats_leather, 'Leather')

htmap_stats_bottle = np.array(htmap_stats_bottle)
det_stats_bottle = np.array(det_stats_bottle)

htmap_stats_hazelnut = np.array(htmap_stats_hazelnut)
det_stats_hazelnut = np.array(det_stats_hazelnut)

htmap_stats_leather = np.array(htmap_stats_leather)
det_stats_leather = np.array(det_stats_leather)

with open(os.path.join(ret_path_bottle, 'eval.txt'), 'w') as f:
    f.write("Explanation - Bottle:\n")
    np.savetxt(f, htmap_stats_bottle, fmt='%s')
    f.write("\nDetection - Bottle:\n")
    np.savetxt(f, det_stats_bottle, fmt='%s')

    f.write("\nExplanation - Hazelnut:\n")
    np.savetxt(f, htmap_stats_hazelnut, fmt='%s')
    f.write("\nDetection - Hazelnut:\n")
    np.savetxt(f, det_stats_hazelnut, fmt='%s')

    f.write("\nExplanation - Leather:\n")
    np.savetxt(f, htmap_stats_leather, fmt='%s')
    f.write("\nDetection - Leather:\n")
    np.savetxt(f, det_stats_leather, fmt='%s')

    print("I dati sono stati salvati nel file eval.txt")


plt.figure(figsize=(12, 6))


plt.subplot(2, 3, 1)
plt.title('Detection')
plt.plot(det_stats_bottle, label='Bottle')
plt.plot(det_stats_hazelnut, label='Hazelnut')
plt.plot(det_stats_leather, label='Leather')
plt.ylim(0.0, 1.1)
plt.xticks(x_values, [str(i) for i in x_values])
#plt.xticks(range(len(det_stats_bottle)), [str(i) for i in range(len(det_stats_bottle))])

plt.xlabel('AUC Score')
plt.ylabel('Budget')
plt.legend(loc='best')
plt.grid(True)

# Grafico per Explanation
plt.subplot(2, 3, 2)
plt.title('Explanation')
plt.plot(htmap_stats_bottle, label='Bottle')
plt.plot(htmap_stats_hazelnut, label='Hazelnut')
plt.plot(htmap_stats_leather, label='Leather')
plt.ylim(0.0, 1.1)
plt.xticks(x_values, [str(i) for i in x_values])
#plt.xticks(range(len(det_stats_bottle)), [str(i) for i in range(len(det_stats_bottle))])

plt.xlabel('AUC Score')
plt.ylabel('Budget')
plt.legend(loc='best')
plt.grid(True)

# Salvataggio e visualizzazione
plt.tight_layout()
plt.savefig(out, bbox_inches='tight')
plt.show()
