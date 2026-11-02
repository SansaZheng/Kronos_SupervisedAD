"""Unknown labels are masked; AP is not trapezoidal PR area."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def choose_threshold(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    mask = y >= 0
    y, probability = y[mask], probability[mask]
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('validation needs both labelled classes to choose threshold')
    p, r, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    # For equal F1 prefer the higher threshold, a deterministic validation-only choice.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(y, probability, threshold):
    y, probability = np.asarray(y), np.asarray(probability)
    mask = y >= 0
    y, probability = y[mask].astype(int), probability[mask]
    if not len(y):
        raise ValueError('no known labels to evaluate')
    tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), positives=int(y.sum()), tp=int(tp), fp=int(fp),
                fn=int(fn), tn=int(tn), precision=float(precision), recall=float(recall),
                f1=float(2 * precision * recall / max(precision + recall, 1e-12)),
                false_positive_rate=float(fp / max(fp + tn, 1)),
                average_precision=float(average_precision_score(y, probability)) if both else None,
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                threshold=float(threshold))
