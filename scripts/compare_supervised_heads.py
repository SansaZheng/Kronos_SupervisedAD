"""Predeclared equal-budget head comparison; finish all validation before any test scoring."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervised.__main__ import parser as training_parser
from supervised.data import check_splits, load
from supervised.engine import file_hash, train, predict


def summarize_metric(values):
    valid = [v for v in values if v is not None]
    return dict(mean=float(np.mean(valid)) if valid else None,
                std=float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
                n=len(valid))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seeds', nargs='+', type=int, default=[13, 42, 2026])
    p.add_argument('--device', default='cpu')
    p.add_argument('--local-only', action='store_true')
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError('comparison needs at least two distinct declared seeds')
    if load(Path(args.data_dir) / 'train.npz').task != 'point':
        raise ValueError('this head comparison is for point tasks; sequence pooling requires its own protocol')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(protocol_version=1, data_dir=args.data_dir, backbone=args.backbone,
        settings=vars(args), variants=['mlp', 'multiscale'],
        shared_training=dict(lr=1e-4, backbone_lr_ratio=.1, layer_decay=1., schedule='cosine',
            warmup_fraction=.1, loss='bce', pos_weight='auto', head_dropout=0., patience=0),
        epoch_selection='validation AP', threshold_selection='validation F1',
        selection_rule='mean validation AP; no selection or tuning from test metrics',
        source_code_sha256={f.name: file_hash(f) for f in sorted(
            (Path(__file__).resolve().parents[1] / 'supervised').glob('*.py'))},
        splits={f'{s}.npz': file_hash(Path(args.data_dir) / f'{s}.npz') for s in ('train', 'validation')})
    # Protocol is persisted before training; test archive is not opened here.
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    reports = []
    for variant in protocol['variants']:
        for seed in args.seeds:
            command = ['train', '--data-dir', args.data_dir, '--backbone', args.backbone,
                '--output', str(root / f'{variant}-{seed}'), '--head-type', variant,
                '--head-dropout', '0', '--sequence-pooling', 'last', '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size), '--patience', '0', '--seed', str(seed),
                '--device', args.device, '--defer-test']
            if args.local_only:
                command.append('--local-only')
            result = train(training_parser().parse_args(command))
            reports.append(dict(variant=variant, seed=seed, output=result['settings']['output'],
                                validation_ap=result['validation']['average_precision']))
    validation_means = {v: float(np.mean([r['validation_ap'] for r in reports if r['variant'] == v]))
                        for v in protocol['variants']}
    selection = dict(validation_mean_ap=validation_means,
                     selected_by_validation=max(validation_means, key=validation_means.get))
    (root / 'validation_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    # All model choices are now fixed. Read and verify the holdout for the first time.
    check_splits([load(Path(args.data_dir) / f'{s}.npz') for s in ('train', 'validation', 'test')])
    for report in reports:
        run = Path(report['output'])
        predict(argparse.Namespace(command='evaluate', checkpoint=str(run / 'best.pt'),
            data=str(Path(args.data_dir) / 'test.npz'), output=str(run / 'test_predictions.csv'),
            device=args.device, batch_size=args.batch_size, threads=2))
        report['test'] = json.loads((run / 'test_predictions.csv.metrics.json').read_text())['metrics']
    summary = dict(protocol=protocol, validation_selection=selection, runs=reports,
                   test_sha256=file_hash(Path(args.data_dir) / 'test.npz'), aggregates={})
    for variant in protocol['variants']:
        selected = [r for r in reports if r['variant'] == variant]
        summary['aggregates'][variant] = {metric: summarize_metric([r['test'][metric] for r in selected])
            for metric in ('average_precision', 'f1', 'precision', 'recall', 'false_positive_rate')}
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(validation_selection=selection, aggregates=summary['aggregates'])), flush=True)


if __name__ == '__main__':
    main()
