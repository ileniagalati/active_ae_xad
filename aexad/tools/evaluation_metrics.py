import numpy as np
from sklearn.metrics import roc_auc_score,jaccard_score, precision_recall_curve

def Xauc(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Area Under the ROC Curve relative to the anomaly score of the pixels of a fixed image
    '''
    return roc_auc_score(ground_truth.flatten(),heatmap.flatten())


def IoU(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: binary matrix containing the anomaly prediction for each pixel
    :return: I/U, where I and U are respectively the intersection and the union of the anomalous pixels of the ground-truth and the predicted heatmap
    '''
    return jaccard_score(ground_truth.flatten(),heatmap.flatten())

def IoU_avg(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Average IoU varying the threshold of the heatmap in order to make it binary
    '''
    summ = 0
    for t in heatmap.flatten():
        binary_heatmap = (heatmap > t).astype(int)
        summ = summ + IoU(ground_truth,binary_heatmap)
    return summ/ground_truth.size

def IoU_max(ground_truth,heatmap):
    '''
    :param ground_truth: binary matrix indicating whether each pixel of an image is actually anomalous (1) or normal (0)
    :param heatmap: real value matrix containing the predicted anomaly score of each pixel
    :return: Maximum IoU varying the threshold of the heatmap in order to make it binary
    '''
    max = 0
    for t in heatmap.flatten():
        binary_heatmap = (heatmap > t).astype(int)
        curr = IoU(ground_truth,binary_heatmap)
        if max<curr:
            max = curr
    return max


def average_precision(y, score):
    precision, _, _ = precision_recall_curve(y, score)
    return precision.mean()

def average_recall(y, score):
    _, recall, _ = precision_recall_curve(y, score)
    return recall.mean()
