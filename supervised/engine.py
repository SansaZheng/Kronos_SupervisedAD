"""Training, validation-selected thresholds, portable checkpoint inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from model import Kronos, KronosTokenizer
from .data import (Standardizer, SequenceDataset, check_splits, load,
                   validate_ohlcv)
from .metrics import select_operating_point, evaluate
from .model import SupervisedKronos, from_specification, tiny_config
from .optimization import labelled_loss, positive_class_weight, make_optimizer, make_scheduler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def option(args, name, default):
    return getattr(args, name, default)


def loader(split, scaler, batch_size, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if shuffle:
        known = split.y >= 0
        if split.task == 'point':
            known &= np.arange(split.X.shape[1])[None] < split.lengths[:, None]
            if split.target_mask is not None:
                known &= split.target_mask
            known = known.any(axis=1)
        eligible = np.flatnonzero(known)
        if not len(eligible):
            raise ValueError('no samples have supervised prediction targets')
        if len(eligible) != len(split.X):
            sampler = SubsetRandomSampler(eligible.tolist(), generator=generator)
            shuffle = False
    return DataLoader(SequenceDataset(split, scaler), batch_size=batch_size,
                      shuffle=shuffle, sampler=sampler, num_workers=0, generator=generator)


@torch.no_grad()
def score(model, batches, device):
    model.eval()
    labels, probabilities, indices, times = [], [], [], []
    for x, y, lengths, sample_indices in batches:
        p = torch.sigmoid(model(x.to(device), lengths.to(device))).cpu()
        if model.task == 'point':
            valid = torch.arange(x.shape[1])[None] < lengths[:, None]
            if batches.dataset.split.target_mask is not None:
                valid &= torch.from_numpy(batches.dataset.split.target_mask[sample_indices.numpy()])
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
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'step', 'label', 'anomaly_probability', 'prediction', 'observation_id'])
        writer.writerows((split.sample_id[i], int(t), int(yy), float(pp), int(pp >= threshold),
                          str(split.observation_id[i, t]) if split.observation_id is not None else '')
                         for yy, pp, i, t in zip(y, p, indices, times))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def isolation_reference(splits):
    reference = dict(sample_id=sorted({str(i) for s in splits for i in s.sample_id}),
                     group_id=None, observation_id=None)
    if splits[0].group_id is not None:
        reference['group_id'] = sorted({str(i) for s in splits for i in s.group_id})
    if splits[0].observation_id is not None:
        reference['observation_id'] = sorted({str(i) for s in splits
            for i in s.observation_id[np.arange(s.X.shape[1])[None] < s.lengths[:, None]]})
    return reference


def check_evaluation_isolation(split, reference):
    if reference is None:
        return False  # legacy checkpoint: cannot retroactively certify unseen data
    for field in ('sample_id', 'group_id', 'observation_id'):
        known = reference[field]
        if known is None:
            continue
        values = getattr(split, field)
        if values is None:
            raise ValueError(f'evaluation needs {field} to verify holdout isolation')
        if field == 'observation_id':
            values = values[np.arange(split.X.shape[1])[None] < split.lengths[:, None]]
        if set(map(str, values)) & set(known):
            raise ValueError(f'evaluation {field} overlaps model training/validation data')
    return True


def save_checkpoint(payload, path):
    temporary = path.with_name('.' + path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_context < 1:
        raise ValueError('epochs, batch size, learning rate and max context must be positive')
    if args.max_context > 2048:
        raise ValueError('max-context cannot exceed the supported Kronos-mini limit of 2048')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if option(args, 'patience', 0) < 0 or option(args, 'min_delta', 0.) < 0:
        raise ValueError('patience and min-delta must be nonnegative')
    if option(args, 'scratch_config', None) and not args.scratch:
        raise ValueError('scratch-config requires explicit --scratch')
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    split_names = ('train', 'validation') if option(args, 'defer_test', False) else ('train', 'validation', 'test')
    paths = [Path(args.data_dir) / f'{s}.npz' for s in split_names]
    splits = [load(p) for p in paths]
    check_splits(splits)
    if any(s.X.shape[1] > args.max_context for s in splits):
        raise ValueError('input exceeds max-context; do not silently truncate labelled sequences')
    train_split, validation = splits[:2]
    test = splits[2] if len(splits) == 3 else None
    labels = train_split.valid_labels()
    if set(np.unique(labels)) != {0, 1} or set(np.unique(validation.valid_labels())) != {0, 1}:
        raise ValueError('training and validation both need known normal and abnormal labels')
    if args.input_kind == 'ohlcv':
        for s in splits:
            validate_ohlcv(s)
    if args.scratch:
        if args.backbone or args.tokenizer or args.input_kind != 'features':
            raise ValueError('scratch is features-only; pretrained OHLCV requires matched weights')
        scratch_config = option(args, 'scratch_config', None)
        config = json.loads(Path(scratch_config).read_text()) if scratch_config else tiny_config()
        backbone, tokenizer = Kronos(**config), None
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
                             args.input_kind, tokenizer, args.freeze_backbone,
                             head_type=option(args, 'head_type', 'mlp'),
                             sequence_pooling=option(args, 'sequence_pooling', 'last'),
                             scales=option(args, 'scales', (4, 16, 64)),
                             head_dropout=option(args, 'head_dropout', 0.)).to(device)
    initial_state_hash = hashlib.sha256()
    for name, tensor in model.backbone.state_dict().items():
        initial_state_hash.update(f'{name}:{tensor.dtype}:{list(tensor.shape)}'.encode())
        initial_state_hash.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    source_code_hashes = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    references = isolation_reference([train_split, validation])
    scaler = Standardizer().fit(train_split)
    train_loader = loader(train_split, scaler, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = loader(validation, scaler, args.batch_size)
    positive_weight = positive_class_weight(labels, option(args, 'pos_weight', 'auto'),
                                             option(args, 'max_pos_weight', None))
    weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    backbone_lr_ratio = option(args, 'backbone_lr_ratio', None)
    if backbone_lr_ratio is None:
        backbone_lr_ratio = 1. if args.scratch else .1
    optimizer, optimizer_groups = make_optimizer(model, args.lr, args.weight_decay,
        backbone_lr_ratio, option(args, 'layer_decay', 1.))
    scheduler = make_scheduler(optimizer, len(train_loader) * args.epochs,
        option(args, 'schedule', 'constant'), option(args, 'warmup_fraction', 0.))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed + 1)  # model size must not silently change shuffle/dropout seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 1)
    history, best_ap, best_epoch, stale = [], -1., None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, known = 0., 0
        for x, y, lengths, _ in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = labelled_loss(model(x, lengths), y, weight,
                                  option(args, 'loss', 'bce'), option(args, 'focal_gamma', 2.))
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            count = int((y >= 0).sum())
            total_loss += float(loss.detach()) * count
            known += count
        val_y, val_p, _, _ = score(model, validation_loader, device)
        metrics = evaluate(val_y, val_p, .5)
        record = dict(epoch=epoch, train_loss=total_loss / max(known, 1),
                      loss_type=option(args, 'loss', 'bce'),
                      validation_average_precision=metrics['average_precision'],
                      learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(record)
        print(json.dumps(record), flush=True)
        if best_epoch is None or metrics['average_precision'] > best_ap + option(args, 'min_delta', 0.):
            best_ap, best_epoch = metrics['average_precision'], epoch
            stale = 0
            save_checkpoint(dict(format_version=2, specification=model.specification(), state_dict=model.state_dict(),
                            evaluation_reference=references,
                            scaler_mean=scaler.mean.tolist(), scaler_scale=scaler.scale.tolist(),
                            feature_names=train_split.feature_names, max_context=args.max_context),
                       output / 'best.pt')
        else:
            stale += 1
        if option(args, 'patience', 0) and stale >= args.patience:
            break
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    val_scores = score(model, validation_loader, device)
    operating_point = select_operating_point(val_scores[0], val_scores[1],
        policy=option(args, 'threshold_policy', 'f1'), max_fpr=option(args, 'max_fpr', .001),
        min_precision=option(args, 'min_precision', .8), beta=option(args, 'beta', 1.))
    threshold = operating_point['threshold']
    checkpoint['threshold'] = threshold
    checkpoint['operating_point'] = operating_point
    save_checkpoint(checkpoint, output / 'best.pt')
    # Test scores are first computed after architecture, epoch and threshold selection.
    test_scores = score(model, loader(test, scaler, args.batch_size), device) if test is not None else None
    report = dict(method='supervised-kronos', task=model.task, input_kind=args.input_kind,
                  initialization=('scratch-config' if option(args, 'scratch_config', None) else 'scratch-tiny')
                                 if args.scratch else str(args.backbone),
                  tokenizer=str(args.tokenizer) if args.tokenizer else None,
                  freeze_backbone=args.freeze_backbone, seed=args.seed,
                  positive_weight=positive_weight, best_epoch=best_epoch,
                  threshold_selection=operating_point,
                  epoch_selection='validation average precision with specified min_delta/patience',
                  epochs_executed=len(history), test_deferred=option(args, 'defer_test', False),
                  model_specification=model.specification(), optimizer_groups=optimizer_groups,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  initial_backbone_state_sha256=initial_state_hash.hexdigest(), source_code_sha256=source_code_hashes,
                  preprocessing='training-only mean imputation, standardization and clip [-5,5]',
                  feature_names=train_split.feature_names,
                  group_isolation_checked=train_split.group_id is not None,
                  observation_isolation_checked=train_split.observation_id is not None,
                  split_archives={p.name: file_hash(p) for p in paths},
                  torch_version=str(torch.__version__), device=str(device),
                  settings=vars(args), history=history,
                  validation=evaluate(val_scores[0], val_scores[1], threshold),
                  test=evaluate(test_scores[0], test_scores[1], threshold) if test_scores is not None else None)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write_predictions(output / 'validation_predictions.csv', validation, val_scores, threshold)
    if test_scores is not None:
        write_predictions(output / 'test_predictions.csv', test, test_scores, threshold)
    print(json.dumps(dict(output=str(output), operating_point=operating_point, test=report['test'])), flush=True)
    return report


def predict(args):
    if args.threads < 1 or args.batch_size < 1:
        raise ValueError('threads and batch-size must be positive')
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    split = load(args.data)
    if split.feature_names != checkpoint['feature_names'] or split.task != checkpoint['specification']['task']:
        raise ValueError('input feature schema/task differs from training checkpoint')
    if split.X.shape[1] > checkpoint['max_context']:
        raise ValueError('input exceeds checkpoint max_context')
    evaluating = option(args, 'command', 'predict') == 'evaluate'
    isolated = check_evaluation_isolation(split, checkpoint.get('evaluation_reference')) if evaluating else False
    output = Path(args.output)
    report_path = output.with_suffix(output.suffix + '.metrics.json')
    if output.exists() or (evaluating and report_path.exists()):
        raise FileExistsError(output)
    if checkpoint['specification']['input_kind'] == 'ohlcv':
        validate_ohlcv(split)
    model = from_specification(checkpoint['specification'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    scaler = Standardizer()
    scaler.mean, scaler.scale = np.asarray(checkpoint['scaler_mean']), np.asarray(checkpoint['scaler_scale'])
    scored = score(model, loader(split, scaler, args.batch_size), torch.device(args.device))
    if evaluating:
        result = dict(metrics=evaluate(scored[0], scored[1], checkpoint['threshold']),
                      holdout_isolation_checked=isolated,
                      operating_point=checkpoint.get('operating_point'),
                      checkpoint_sha256=file_hash(args.checkpoint), input_sha256=file_hash(args.data))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(output, split, scored, checkpoint['threshold'])
    if evaluating:
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), predictions=len(scored[1]))))
