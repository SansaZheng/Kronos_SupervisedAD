"""Explicit split archives, train-only preprocessing, and publisher-labelled data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OHLCVA = ['open', 'high', 'low', 'close', 'volume', 'amount']
TARGET_COLUMNS = {'gt', 'class', 'label', 'target', 'y', 'isfraud', 'txfraud',
                  'islaundering', 'launderingtype', 'txfraudscenario'}


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    lengths: np.ndarray
    sample_id: np.ndarray
    feature_names: list[str]
    group_id: np.ndarray | None = None

    @property
    def task(self):
        return 'sequence' if self.y.ndim == 1 else 'point'

    def valid_labels(self):
        if self.task == 'sequence':
            return self.y[self.y >= 0]
        mask = np.arange(self.X.shape[1])[None] < self.lengths[:, None]
        return self.y[mask & (self.y >= 0)]


def validate(split):
    x, y = split.X, split.y
    if x.ndim != 3 or not len(x) or not x.shape[1] or not x.shape[2]:
        raise ValueError('X must be nonempty [N,T,F]')
    n, t, f = x.shape
    if y.shape not in ((n,), (n, t)) or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError('y must be [N] or [N,T], with labels -1/0/1 only')
    if split.lengths.shape != (n,) or not np.issubdtype(split.lengths.dtype, np.integer):
        raise ValueError('lengths must be integer [N]')
    if np.any(split.lengths < 1) or np.any(split.lengths > t):
        raise ValueError('lengths outside 1..T')
    if len(split.feature_names) != f or len(set(split.feature_names)) != f:
        raise ValueError('feature_names must be unique and match F')
    canonical_names = {''.join(c for c in name.lower() if c.isalnum()) for name in split.feature_names}
    if canonical_names & TARGET_COLUMNS:
        raise ValueError('target/label columns must not appear in feature_names')
    if split.sample_id.shape != (n,) or len(set(split.sample_id.tolist())) != n:
        raise ValueError('sample_id must be unique [N]')
    if split.group_id is not None and split.group_id.shape != (n,):
        raise ValueError('group_id must have shape [N]')
    valid = np.arange(t)[None] < split.lengths[:, None]
    if np.isinf(x[valid]).any():
        raise ValueError('infinite features are not supported; NaNs are train-imputed')
    return split


def load(path):
    with np.load(path, allow_pickle=False) as z:
        for field in ('X', 'y', 'sample_id', 'feature_names'):
            if field not in z:
                raise ValueError(f'missing NPZ field: {field}')
        x = z['X'].astype(np.float32)
        return validate(Split(x, z['y'], z['lengths'] if 'lengths' in z else
                              np.full(len(x), x.shape[1], dtype=np.int64),
                              z['sample_id'].astype(str), z['feature_names'].astype(str).tolist(),
                              z['group_id'].astype(str) if 'group_id' in z else None))


def save(path, split):
    validate(split)
    payload = dict(X=split.X, y=split.y, lengths=split.lengths,
                   sample_id=split.sample_id, feature_names=np.asarray(split.feature_names))
    if split.group_id is not None:
        payload['group_id'] = split.group_id
    np.savez_compressed(path, **payload)


def check_splits(splits):
    """Reject overlapping IDs/groups; ungrouped sequences also require deduplication.

    Equal features on independent known groups are legitimate observations (in
    particular sparse background trading blocks), not proof of shared source rows.
    """
    if any(s.task != splits[0].task or s.feature_names != splits[0].feature_names for s in splits):
        raise ValueError('all splits must share task and ordered feature schema')
    if any((s.group_id is None) != (splits[0].group_id is None) for s in splits):
        raise ValueError('group_id must be supplied consistently across splits')
    ids, groups, hashes = set(), set(), set()
    for s in splits:
        current = set(s.sample_id.tolist())
        current_groups = set(s.group_id.tolist()) if s.group_id is not None else set()
        current_hashes = ({hashlib.sha256(np.asarray(x[:length], dtype='<f4').tobytes()).hexdigest()
                           for x, length in zip(s.X, s.lengths)} if s.group_id is None else set())
        if ids & current or groups & current_groups or hashes & current_hashes:
            raise ValueError('cross-split sample/group/content overlap')
        ids |= current
        groups |= current_groups
        hashes |= current_hashes


class Standardizer:
    """Training observations only; no per-full-window future normalization."""
    def fit(self, split):
        valid = np.arange(split.X.shape[1])[None] < split.lengths[:, None]
        values = split.X[valid].astype(np.float64)
        self.mean = np.nanmean(values, axis=0)
        if not np.isfinite(self.mean).all():
            raise ValueError('a training feature is entirely missing')
        values = np.where(np.isnan(values), self.mean, values)
        self.scale = np.maximum(values.std(axis=0), 1e-6)
        return self

    def transform(self, x):
        return np.clip((np.where(np.isnan(x), self.mean, x) - self.mean) / self.scale,
                       -5., 5.).astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, split, scaler):
        self.split = split
        self.x = scaler.transform(split.X)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        length = int(self.split.lengths[i])
        y = np.array(self.split.y[i], dtype=np.float32, copy=True)
        if y.ndim:
            y[length:] = -1
        return torch.from_numpy(self.x[i]), torch.tensor(y), length, i


def validate_ohlcv(split):
    if split.feature_names != OHLCVA:
        raise ValueError('ohlcv feature_names must be open,high,low,close,volume,amount in that order')
    mask = np.arange(split.X.shape[1])[None] < split.lengths[:, None]
    x = split.X[mask]
    if not np.isfinite(x).all() or (x[:, :4] <= 0).any() or (x[:, 4:] < 0).any():
        raise ValueError('OHLCVA must be finite with positive prices and nonnegative volume/amount')
    if (x[:, 1] < x[:, [0, 2, 3]].max(axis=1)).any() or (x[:, 2] > x[:, [0, 1, 3]].min(axis=1)).any():
        raise ValueError('invalid OHLC price ordering')


def prepare_builtin(dataset, source_root, output, chunk_length=128):
    from scripts.data.load_supervised_data import load_sapienza, load_sd_fmm, load_tstr
    from scripts.data.prepare_supervised_data import FEATURES
    if chunk_length < 1:
        raise ValueError('chunk length must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    ulb = None
    if dataset == 'ulb':
        from scripts.data.transaction_data import iter_table
        ulb = pd.concat(iter_table('ulb', root=Path(source_root) / 'transactions'), ignore_index=True)
        ulb['_source_row'] = np.arange(len(ulb))
        ulb = ulb.sort_values('Time', kind='stable').reset_index(drop=True)
        # Move boundaries to the beginning of a timestamp tie, preserving strict time separation.
        bounds = [0] + [int(ulb.Time.searchsorted(ulb.Time.iloc[int(len(ulb) * fraction)], side='left'))
                        for fraction in (.7, .85)] + [len(ulb)]
    for split_index, name in enumerate(('train', 'validation', 'test')):
        if dataset == 'sapienza':
            x, y, meta = load_sapienza('15S', name, source_root)
            # Event-local observed blocks, without filling missing clock intervals.
            # The gap is available at the current observation and is a causal feature.
            meta = meta.copy()
            meta['_row'] = np.arange(len(meta))
            chunks, labels, lengths, ids, group_ids = [], [], [], [], []
            for event, frame in meta.groupby('pump_index', sort=True):
                frame = frame.sort_values('date', kind='stable')
                times = pd.to_datetime(frame.date, errors='raise')
                gaps = np.log1p(times.diff().dt.total_seconds().fillna(0).to_numpy())
                indices = frame['_row'].to_numpy()
                event_x = np.column_stack([x[indices], gaps])
                for start in range(0, len(indices), chunk_length):
                    rows = indices[start:start + chunk_length]
                    length = len(rows)
                    xp = np.zeros((chunk_length, x.shape[-1] + 1), dtype=np.float32)
                    yp = np.full(chunk_length, -1, dtype=np.int64)
                    xp[:length], yp[:length] = event_x[start:start + length], y[rows]
                    chunks.append(xp); labels.append(yp); lengths.append(length)
                    ids.append(f'sapienza:{event}:{int(rows[0])}')
                    group_ids.append(str(frame.symbol.iloc[0]))
            split = Split(np.asarray(chunks), np.asarray(labels), np.asarray(lengths),
                          np.asarray(ids), list(FEATURES) + ['log1p_gap_seconds'], np.asarray(group_ids))
        elif dataset == 'ulb':
            frame = ulb.iloc[bounds[split_index]:bounds[split_index + 1]]
            features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            chunks, labels, lengths, ids = [], [], [], []
            for start in range(0, len(frame), chunk_length):
                rows = frame.iloc[start:start + chunk_length]
                xp = np.zeros((chunk_length, len(features)), dtype=np.float32)
                yp = np.full(chunk_length, -1, dtype=np.int64)
                xp[:len(rows)] = rows[features].to_numpy(dtype=np.float32)
                yp[:len(rows)] = rows.Class.to_numpy(dtype=np.int64)
                chunks.append(xp); labels.append(yp); lengths.append(len(rows))
                ids.append(f'ulb:{int(rows._source_row.iloc[0])}:{int(rows._source_row.iloc[-1])}')
            split = Split(np.asarray(chunks), np.asarray(labels), np.asarray(lengths),
                          np.asarray(ids), features)
        elif dataset == 'tstr':
            x, y, ids = load_tstr(name, source_root)
            split = Split(x.astype(np.float32), y, np.full(len(x), x.shape[1]),
                          ids.astype(str), ['High'])
        else:
            market = dataset.removeprefix('sd-')
            x, y, indices = load_sd_fmm(market, name, source_root)
            split = Split(x.astype(np.float32), y, np.full(len(x), x.shape[1]),
                          np.asarray([f'sd-{market}:{i}' for i in indices]), ['processed_price'])
        save(output / f'{name}.npz', split)
        known = split.valid_labels()
        summaries[name] = dict(sequences=len(split.X), valid_observations=int(split.lengths.sum()),
                               positives=int((known == 1).sum()), negatives=int((known == 0).sum()),
                               feature_names=split.feature_names)
    check_splits([load(output / f'{s}.npz') for s in ('train', 'validation', 'test')])
    report = dict(dataset=dataset, source_root=str(source_root), chunk_length=chunk_length,
                  split_policy=('chronological 70/15/15 percent, timestamp ties kept together; global stream, customer IDs unavailable'
                                if dataset == 'ulb' else 'existing repository research splits; no relabelling'),
                  summaries=summaries)
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
