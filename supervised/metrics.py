"""Unknown labels are masked; AP is not trapezoidal PR area."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def known_scores(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    if y.ndim != 1 or y.shape != probability.shape or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('scores require equal-length 1D labels -1/0/1')
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('probabilities must be finite in [0,1]')
    mask = y >= 0
    return y[mask].astype(int), probability[mask]


def select_operating_point(y, probability, policy='f1', max_fpr=.001, min_precision=.8, beta=1.):
    """Validation-only operating point; budget is empirical, not a test guarantee.

    Tied scores are indivisible, matching the deployed p>=threshold decision.
    A no-alert candidate permits strict budgets; precision constraints with no
    nonempty feasible alert set explicitly report feasible=False.
    """
    y, probability = known_scores(y, probability)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('validation needs both labelled classes to choose threshold')
    if (policy not in ('f1', 'fpr', 'precision', 'fbeta') or not 0 <= max_fpr <= 1
            or not 0 <= min_precision <= 1 or not np.isfinite(beta) or beta <= 0):
        raise ValueError('invalid threshold policy, budget or beta')
    order = np.argsort(-probability, kind='stable')
    sorted_p, sorted_y = probability[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(y) - 1]
    tp = np.r_[0, sorted_y.cumsum()[ends]]
    fp = np.r_[0, ends + 1 - sorted_y.cumsum()[ends]]
    thresholds = np.r_[np.nextafter(float(sorted_p[0]), np.inf), sorted_p[ends]]
    precision = tp / np.maximum(tp + fp, 1)
    recall, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    if policy in ('f1', 'fbeta'):
        b2 = 1. if policy == 'f1' else beta ** 2
        criterion = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
        candidates = np.flatnonzero(criterion == criterion.max())
        feasible = True
    else:
        valid = fpr <= max_fpr if policy == 'fpr' else ((precision >= min_precision) & (tp + fp > 0))
        feasible = bool(valid.any())
        if not feasible:
            candidates = np.array([0])
        else:
            candidates = np.flatnonzero(valid & (recall == recall[valid].max()))
            candidates = candidates[precision[candidates] == precision[candidates].max()]
    chosen = candidates[np.argmax(thresholds[candidates])]
    return dict(threshold=float(thresholds[chosen]), policy=policy, feasible=feasible,
                status=('precision_unattainable' if not feasible else
                        'no_alerts' if tp[chosen] + fp[chosen] == 0 else 'active'),
                validation_precision=float(precision[chosen]), validation_recall=float(recall[chosen]),
                validation_fpr=float(fpr[chosen]), validation_alerts=int(tp[chosen] + fp[chosen]),
                validation_normals=int((y == 0).sum()), validation_positives=int(y.sum()),
                max_fpr=max_fpr if policy == 'fpr' else None,
                min_precision=min_precision if policy == 'precision' else None,
                beta=beta if policy == 'fbeta' else 1.)


def choose_threshold(y, probability, **kwargs):
    return select_operating_point(y, probability, **kwargs)['threshold']


def evaluate(y, probability, threshold):
    y, probability = known_scores(y, probability)
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
                false_alerts_per_1000=float(1000 * fp / len(y)),
                prevalence=float(y.mean()),
                brier_score=float(np.mean((probability - y) ** 2)),
                average_precision=float(average_precision_score(y, probability)) if both else None,
                roc_auc=float(roc_auc_score(y, probability)) if both else None,
                threshold=float(threshold))
