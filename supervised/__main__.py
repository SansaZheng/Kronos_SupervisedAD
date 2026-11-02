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
    train = commands.add_parser('train', help='Fit labelled BCE; select epoch/threshold on validation only')
    train.add_argument('--data-dir', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--backbone', help='HF model ID or local Kronos weight directory')
    train.add_argument('--tokenizer', help='Matched tokenizer; required only for genuine OHLCVA')
    train.add_argument('--scratch', action='store_true', help='Explicit tiny randomly initialized architecture ablation')
    train.add_argument('--input-kind', choices=['features', 'ohlcv'], default='features')
    train.add_argument('--freeze-backbone', action='store_true')
    train.add_argument('--local-only', action='store_true')
    train.add_argument('--max-context', type=int, default=512)
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--batch-size', type=int, default=16)
    train.add_argument('--lr', type=float, default=1e-4)
    train.add_argument('--weight-decay', type=float, default=.01)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--device', default='cpu')
    train.add_argument('--threads', type=int, default=2)
    predict = commands.add_parser('predict', help='Offline checkpoint inference using stored threshold')
    predict.add_argument('--checkpoint', required=True)
    predict.add_argument('--data', required=True, help='Same NPZ schema; y may be entirely -1')
    predict.add_argument('--output', required=True)
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--batch-size', type=int, default=16)
    predict.add_argument('--threads', type=int, default=2)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'prepare':
        prepare_builtin(args.dataset, args.source_root, args.output, args.chunk_length)
    else:
        from .engine import train, predict
        (train if args.command == 'train' else predict)(args)


if __name__ == '__main__':
    main()
