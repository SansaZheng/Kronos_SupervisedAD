"""Training, validation-selected thresholds, portable checkpoint inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import Kronos, KronosTokenizer
from .data import (Standardizer, SequenceDataset, check_splits, load,
                   validate_ohlcv)
from .metrics import choose_threshold, evaluate
from .model import SupervisedKronos, from_specification, tiny_config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labelled_loss(logits, labels, positive_weight):
    mask = labels >= 0
    if not mask.any():
        return None
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask],
                                              pos_weight=positive_weight)


def loader(split, scaler, batch_size, shuffle=False):
    return DataLoader(SequenceDataset(split, scaler), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


@torch.no_grad()
def score(model, batches, device):
    model.eval()
    labels, probabilities, indices, times = [], [], [], []
    for x, y, lengths, sample_indices in batches:
        p = torch.sigmoid(model(x.to(device), lengths.to(device))).cpu()
        if model.task == 'point':
            valid = torch.arange(x.shape[1])[None] < lengths[:, None]
            yy, tt = torch.where(valid)
            labels.extend(y[valid].tolist())
            probabilities.extend(p[valid].tolist())
            indices.extend(sample_indices[yy].tolist())
            times.extend(tt.tolist())
        else:
            labels.extend(y.tolist()); probabilities.extend(p.tolist())
            indices.extend(sample_indices.tolist()); times.extend([-1] * len(x))
    return (np.asarray(labels), np.asarray(probabilities),
            np.asarray(indices), np.asarray(times))


def write_predictions(path, split, scored, threshold):
    y, p, indices, times = scored
    with Path(path).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'step', 'label', 'anomaly_probability', 'prediction'])
        writer.writerows((split.sample_id[i], int(t), int(yy), float(pp), int(pp >= threshold))
                         for yy, pp, i, t in zip(y, p, indices, times))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_context < 1:
        raise ValueError('epochs, batch size, learning rate and max context must be positive')
    if args.max_context > 2048:
        raise ValueError('max-context cannot exceed the supported Kronos-mini limit of 2048')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    paths = [Path(args.data_dir) / f'{s}.npz' for s in ('train', 'validation', 'test')]
    splits = [load(p) for p in paths]
    check_splits(splits)
    if any(s.X.shape[1] > args.max_context for s in splits):
        raise ValueError('input exceeds max-context; do not silently truncate labelled sequences')
    train_split, validation, test = splits
    labels = train_split.valid_labels()
    if set(np.unique(labels)) != {0, 1} or set(np.unique(validation.valid_labels())) != {0, 1}:
        raise ValueError('training and validation both need known normal and abnormal labels')
    if args.input_kind == 'ohlcv':
        for s in splits:
            validate_ohlcv(s)
    if args.scratch:
        if args.backbone or args.tokenizer or args.input_kind != 'features':
            raise ValueError('scratch is features-only; pretrained OHLCV requires matched weights')
        backbone, tokenizer = Kronos(**tiny_config()), None
    else:
        if not args.backbone:
            raise ValueError('provide --backbone or explicitly select --scratch')
        if args.max_context > 512 and 'kronos-mini' not in str(args.backbone).lower():
            raise ValueError('only Kronos-mini is supported above 512 context positions')
        backbone = Kronos.from_pretrained(args.backbone, local_files_only=args.local_only)
        tokenizer = (KronosTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_only)
                     if args.input_kind == 'ohlcv' and args.tokenizer else None)
        if args.input_kind == 'features' and args.tokenizer:
            raise ValueError('feature adapter bypasses the tokenizer; omit --tokenizer')
    model = SupervisedKronos(backbone, train_split.X.shape[-1], train_split.task,
                             args.input_kind, tokenizer, args.freeze_backbone).to(device)
    scaler = Standardizer().fit(train_split)
    train_loader = loader(train_split, scaler, args.batch_size, shuffle=True)
    validation_loader = loader(validation, scaler, args.batch_size)
    positive_weight = float((labels == 0).sum() / (labels == 1).sum())
    weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    history, best_ap, best_epoch = [], -1., None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, known = 0., 0
        for x, y, lengths, _ in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = labelled_loss(model(x, lengths), y, weight)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            count = int((y >= 0).sum())
            total_loss += float(loss.detach()) * count
            known += count
        val_y, val_p, _, _ = score(model, validation_loader, device)
        metrics = evaluate(val_y, val_p, .5)
        record = dict(epoch=epoch, train_weighted_bce=total_loss / max(known, 1),
                      validation_average_precision=metrics['average_precision'])
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics['average_precision'] > best_ap:
            best_ap, best_epoch = metrics['average_precision'], epoch
            torch.save(dict(specification=model.specification(), state_dict=model.state_dict(),
                            scaler_mean=scaler.mean.tolist(), scaler_scale=scaler.scale.tolist(),
                            feature_names=train_split.feature_names, max_context=args.max_context),
                       output / 'best.pt')
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    val_scores = score(model, validation_loader, device)
    threshold = choose_threshold(val_scores[0], val_scores[1])
    checkpoint['threshold'] = threshold
    torch.save(checkpoint, output / 'best.pt')
    # Test scores are first computed after architecture, epoch and threshold selection.
    test_scores = score(model, loader(test, scaler, args.batch_size), device)
    report = dict(method='supervised-kronos', task=model.task, input_kind=args.input_kind,
                  initialization='scratch-tiny' if args.scratch else str(args.backbone),
                  tokenizer=str(args.tokenizer) if args.tokenizer else None,
                  freeze_backbone=args.freeze_backbone, seed=args.seed,
                  positive_weight=positive_weight, best_epoch=best_epoch,
                  threshold_selection='validation F1; highest threshold breaks ties',
                  epoch_selection='validation average precision',
                  preprocessing='training-only mean imputation, standardization and clip [-5,5]',
                  feature_names=train_split.feature_names,
                  group_isolation_checked=train_split.group_id is not None,
                  split_archives={p.name: file_hash(p) for p in paths},
                  torch_version=str(torch.__version__), device=str(device),
                  settings=vars(args), history=history,
                  validation=evaluate(val_scores[0], val_scores[1], threshold),
                  test=evaluate(test_scores[0], test_scores[1], threshold))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_predictions(output / 'validation_predictions.csv', validation, val_scores, threshold)
    write_predictions(output / 'test_predictions.csv', test, test_scores, threshold)
    print(json.dumps(dict(output=str(output), test=report['test'])), flush=True)
    return report


def predict(args):
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    split = load(args.data)
    if split.feature_names != checkpoint['feature_names'] or split.task != checkpoint['specification']['task']:
        raise ValueError('input feature schema/task differs from training checkpoint')
    if split.X.shape[1] > checkpoint['max_context']:
        raise ValueError('input exceeds checkpoint max_context')
    if checkpoint['specification']['input_kind'] == 'ohlcv':
        validate_ohlcv(split)
    model = from_specification(checkpoint['specification'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    scaler = Standardizer()
    scaler.mean, scaler.scale = np.asarray(checkpoint['scaler_mean']), np.asarray(checkpoint['scaler_scale'])
    scored = score(model, loader(split, scaler, args.batch_size), torch.device(args.device))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, split, scored, checkpoint['threshold'])
    print(json.dumps(dict(output=str(output), predictions=len(scored[1]))))
