import numpy as np

def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    return (2. * intersection + eps) / (pred.sum() + gt.sum() + eps)

def precision_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    return (tp + eps) / (tp + fn + eps)
