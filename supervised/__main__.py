"""python -m supervised prepare|train|predict"""
import argparse

from .data import prepare_builtin


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Preserve existing publisher-labelled research splits')
    prepare.add_argument('--dataset', choices=['sapienza', 'tstr', 'sd-stock', 'sd-crypto', 'ulb'], required=True)
    prepare.add_argument('--source-root', default='data/supervised_financial')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--chunk-length', type=int, default=128)
    prepare.add_argument('--history-length', type=int, default=0,
                         help='Past-only overlap for Sapienza/ULB; history is not a target')
    train = commands.add_parser('train', help='Fit labelled BCE; select epoch/threshold on validation only')
    train.add_argument('--data-dir', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--backbone', help='HF model ID or local Kronos weight directory')
    train.add_argument('--tokenizer', help='Matched tokenizer; required only for genuine OHLCVA')
    train.add_argument('--scratch', action='store_true', help='Explicit tiny randomly initialized architecture ablation')
    train.add_argument('--scratch-config', help='JSON Kronos config for architecture-matched scratch control')
    train.add_argument('--input-kind', choices=['features', 'ohlcv'], default='features')
    train.add_argument('--freeze-backbone', action='store_true')
    train.add_argument('--local-only', action='store_true')
    train.add_argument('--max-context', type=int, default=512)
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--batch-size', type=int, default=16)
    train.add_argument('--lr', type=float, default=1e-4)
    train.add_argument('--weight-decay', type=float, default=.01)
    train.add_argument('--head-type', choices=['mlp', 'multiscale'], default='multiscale')
    train.add_argument('--sequence-pooling', choices=['last', 'attention'], default='attention')
    train.add_argument('--scales', nargs='+', type=int, default=[4, 16, 64])
    train.add_argument('--head-dropout', type=float, default=.1)
    train.add_argument('--loss', choices=['bce', 'focal'], default='bce')
    train.add_argument('--focal-gamma', type=float, default=2.)
    train.add_argument('--pos-weight', default='auto', help='auto, sqrt, none, or a positive number; training only')
    train.add_argument('--max-pos-weight', type=float)
    train.add_argument('--backbone-lr-ratio', type=float,
                       help='Default .1 for pretrained backbone, 1 for scratch')
    train.add_argument('--layer-decay', type=float, default=1.)
    train.add_argument('--schedule', choices=['constant', 'cosine'], default='cosine')
    train.add_argument('--warmup-fraction', type=float, default=.1)
    train.add_argument('--patience', type=int, default=3, help='Validation AP patience; 0 disables early stopping')
    train.add_argument('--min-delta', type=float, default=0.)
    train.add_argument('--threshold-policy', choices=['f1', 'fbeta', 'fpr', 'precision'], default='f1')
    train.add_argument('--max-fpr', type=float, default=.001, help='Empirical validation FPR budget for fpr policy')
    train.add_argument('--min-precision', type=float, default=.8)
    train.add_argument('--beta', type=float, default=1.)
    train.add_argument('--defer-test', action='store_true', help='Do not open test.npz; evaluate frozen checkpoint later')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', default='cpu')
    train.add_argument('--threads', type=int, default=2)
    for command in ('predict', 'evaluate'):
        inference = commands.add_parser(command, help='Offline inference/evaluation using the stored fixed threshold')
        inference.add_argument('--checkpoint', required=True)
        inference.add_argument('--data', required=True, help='Same NPZ schema; predict allows all y=-1')
        inference.add_argument('--output', required=True, help='Prediction CSV; evaluate also writes .metrics.json')
        inference.add_argument('--device', default='cpu')
        inference.add_argument('--batch-size', type=int, default=16)
        inference.add_argument('--threads', type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        prepare_builtin(args.dataset, args.source_root, args.output, args.chunk_length, args.history_length)
    else:
        from .engine import train, predict
        (train if args.command == 'train' else predict)(args)


if __name__ == '__main__':
    main()
