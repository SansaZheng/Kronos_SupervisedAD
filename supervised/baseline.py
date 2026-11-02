"""A fixed boosting control using the same labelled splits and causal raw features."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Standardizer, check_splits, load
from .engine import file_hash, write_predictions
from .metrics import evaluate, select_operating_point


def temporal_features(x, scales=(4, 16, 64)):
    """Current values, one-step change, and deviations from trailing means."""
    delta = x - np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    prefix = np.concatenate([np.zeros_like(x[:, :1], dtype=np.float64),
                             x.cumsum(axis=1, dtype=np.float64)], axis=1)
    ends = np.arange(1, x.shape[1] + 1)
    features = [x, delta]
    for width in scales:
        starts = np.maximum(ends - width, 0)
        mean = (prefix[:, ends] - prefix[:, starts]) / (ends - starts)[None, :, None]
        features.append(x - mean)
    return np.concatenate(features, axis=-1).astype(np.float32)


def design(split, scaler):
    x = scaler.transform(split.X)
    if split.task == 'point':
        features = temporal_features(x)
        valid = np.arange(x.shape[1])[None] < split.lengths[:, None]
        if split.target_mask is not None:
            valid &= split.target_mask
        indices, times = np.where(valid)
        return features[valid], split.y[valid], indices, times
    rows = []
    for series, length in zip(x, split.lengths):
        valid = series[:length]
        rows.append(np.concatenate([valid[-1], valid.mean(0), valid.std(0),
                                     valid.min(0), valid.max(0), valid[-1] - valid[0]]))
    return np.asarray(rows), split.y, np.arange(len(x)), np.full(len(x), -1)


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    names = ['train', 'validation'] if args.defer_test else ['train', 'validation', 'test']
    paths = [Path(args.data_dir) / f'{name}.npz' for name in names]
    splits = [load(p) for p in paths]
    check_splits(splits)
    scaler = Standardizer().fit(splits[0])
    x, y, _, _ = design(splits[0], scaler)
    known = y >= 0
    if set(np.unique(y[known])) != {0, 1}:
        raise ValueError('baseline training needs both classes')
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
        l2_regularization=1., class_weight='balanced', early_stopping=False, random_state=args.seed)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        model.fit(x[known], y[known])
        vx, vy, vi, vt = design(splits[1], scaler)
        vp = model.predict_proba(vx)[:, 1]
        op = select_operating_point(vy, vp, policy=args.threshold_policy,
                                    max_fpr=args.max_fpr, min_precision=args.min_precision, beta=args.beta)
        output.mkdir(parents=True, exist_ok=False)
        write_predictions(output / 'validation_predictions.csv', splits[1], (vy, vp, vi, vt), op['threshold'])
        test_metrics = None
        if not args.defer_test:
            tx, ty, ti, tt = design(splits[2], scaler)
            tp = model.predict_proba(tx)[:, 1]
            test_metrics = evaluate(ty, tp, op['threshold'])
            write_predictions(output / 'test_predictions.csv', splits[2], (ty, tp, ti, tt), op['threshold'])
    import joblib
    joblib.dump(dict(model=model, mean=scaler.mean, scale=scaler.scale, threshold=op['threshold'],
                     feature_names=splits[0].feature_names, task=splits[0].task), output / 'model.joblib')
    report = dict(method='hist-gradient-boosting', seed=args.seed, settings=vars(args),
                  features='point: current/change/trailing deviations 4,16,64; sequence: whole valid summary',
                  parameters=model.get_params(), threshold_selection=op,
                  split_archives={p.name: file_hash(p) for p in paths},
                  validation=evaluate(vy, vp, op['threshold']), test=test_metrics)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--threshold-policy', choices=['f1', 'fbeta', 'fpr', 'precision'], default='f1')
    p.add_argument('--max-fpr', type=float, default=.001)
    p.add_argument('--min-precision', type=float, default=.8)
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--defer-test', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
