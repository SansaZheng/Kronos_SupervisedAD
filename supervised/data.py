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
    target_mask: np.ndarray | None = None
    observation_id: np.ndarray | None = None

    @property
    def task(self):
        return 'sequence' if self.y.ndim == 1 else 'point'

    def valid_labels(self):
        if self.task == 'sequence':
            return self.y[self.y >= 0]
        mask = np.arange(self.X.shape[1])[None] < self.lengths[:, None]
        if self.target_mask is not None:
            mask &= self.target_mask
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
    if split.target_mask is not None:
        if (split.task != 'point' or split.target_mask.shape != (n, t)
                or split.target_mask.dtype != np.bool_):
            raise ValueError('target_mask must be boolean [N,T] for point tasks')
        if (split.target_mask & ~valid).any() or not split.target_mask.any(axis=1).all():
            raise ValueError('targets must be valid observations, with at least one per sample')
    if split.observation_id is not None:
        if split.task != 'point' or split.observation_id.shape != (n, t):
            raise ValueError('observation_id must have shape [N,T] for point tasks')
        if (split.observation_id[valid].astype(str) == '').any():
            raise ValueError('valid observations require nonempty observation_id')
        targets = valid if split.target_mask is None else split.target_mask
        ids = split.observation_id[targets].astype(str)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('each observation_id may be a prediction target only once')
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
                              z['group_id'].astype(str) if 'group_id' in z else None,
                              z['target_mask'] if 'target_mask' in z else None,
                              z['observation_id'].astype(str) if 'observation_id' in z else None))


def save(path, split):
    validate(split)
    payload = dict(X=split.X, y=split.y, lengths=split.lengths,
                   sample_id=split.sample_id, feature_names=np.asarray(split.feature_names))
    if split.group_id is not None:
        payload['group_id'] = split.group_id
    if split.target_mask is not None:
        payload['target_mask'] = split.target_mask
    if split.observation_id is not None:
        payload['observation_id'] = split.observation_id
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
    if any((s.observation_id is None) != (splits[0].observation_id is None) for s in splits):
        raise ValueError('observation_id must be supplied consistently across splits')
    ids, groups, hashes, observations = set(), set(), set(), set()
    for s in splits:
        current = set(s.sample_id.tolist())
        current_groups = set(s.group_id.tolist()) if s.group_id is not None else set()
        current_hashes = ({hashlib.sha256(np.asarray(x[:length], dtype='<f4').tobytes()).hexdigest()
                           for x, length in zip(s.X, s.lengths)} if s.group_id is None else set())
        valid = np.arange(s.X.shape[1])[None] < s.lengths[:, None]
        current_observations = set(s.observation_id[valid].astype(str)) if s.observation_id is not None else set()
        if ids & current or groups & current_groups or hashes & current_hashes or observations & current_observations:
            raise ValueError('cross-split sample/group/content overlap')
        ids |= current
        groups |= current_groups
        hashes |= current_hashes
        observations |= current_observations


class Standardizer:
    """Training observations only; no per-full-window future normalization."""
    def fit(self, split):
        valid = np.arange(split.X.shape[1])[None] < split.lengths[:, None]
        if split.target_mask is not None:
            valid &= split.target_mask  # reused history must not overweight preprocessing statistics
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
            if self.split.target_mask is not None:
                y[~self.split.target_mask[i]] = -1
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


def chunk_point_stream(x, y, observation_ids, chunk_length, history_length, prefix, feature_names, group=None):
    """Reuse past context, but supervise/export each original observation exactly once."""
    if chunk_length < 1 or history_length < 0 or not len(x):
        raise ValueError('invalid chunk/history length or empty stream')
    width = chunk_length + history_length
    n = (len(x) + chunk_length - 1) // chunk_length
    values = np.zeros((n, width, x.shape[-1]), dtype=np.float32)
    labels = np.full((n, width), -1, dtype=np.int64)
    target_mask = np.zeros((n, width), dtype=bool)
    ids = np.full((n, width), '', dtype=f'<U{max(len(str(i)) for i in observation_ids)}')
    lengths, sample_ids = [], []
    for i, start in enumerate(range(0, len(x), chunk_length)):
        begin, end = max(0, start - history_length), min(len(x), start + chunk_length)
        length, offset = end - begin, start - begin
        values[i, :length] = x[begin:end]
        labels[i, offset:length] = y[start:end]
        target_mask[i, offset:length] = True
        ids[i, :length] = observation_ids[begin:end]
        lengths.append(length)
        sample_ids.append(f'{prefix}:{start}')
    return Split(values, labels, np.asarray(lengths), np.asarray(sample_ids), feature_names,
                 np.asarray([group] * n) if group is not None else None, target_mask, ids)


def prepare_builtin(dataset, source_root, output, chunk_length=128, history_length=0):
    from scripts.data.load_supervised_data import load_sapienza, load_sd_fmm, load_tstr
    from scripts.data.prepare_supervised_data import FEATURES
    if chunk_length < 1 or history_length < 0:
        raise ValueError('chunk length must be positive and history length nonnegative')
    if history_length and dataset not in ('sapienza', 'ulb'):
        raise ValueError('history-length applies only to chunked point streams, not complete labelled sequences')
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
            streams = []
            for event, frame in meta.groupby('pump_index', sort=True):
                frame = frame.sort_values('date', kind='stable')
                times = pd.to_datetime(frame.date, errors='raise')
                gaps = np.log1p(times.diff().dt.total_seconds().fillna(0).to_numpy())
                indices = frame['_row'].to_numpy()
                event_x = np.column_stack([x[indices], gaps])
                streams.append(chunk_point_stream(event_x, y[indices],
                    [f'sapienza:{event}:{i}' for i in range(len(indices))], chunk_length,
                    history_length, f'sapienza:{event}', list(FEATURES) + ['log1p_gap_seconds'],
                    str(frame.symbol.iloc[0])))
            split = Split(*(np.concatenate([getattr(s, field) for s in streams])
                            for field in ('X', 'y', 'lengths', 'sample_id')),
                          streams[0].feature_names,
                          *(np.concatenate([getattr(s, field) for s in streams])
                            for field in ('group_id', 'target_mask', 'observation_id')))
        elif dataset == 'ulb':
            frame = ulb.iloc[bounds[split_index]:bounds[split_index + 1]]
            features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            split = chunk_point_stream(frame[features].to_numpy(dtype=np.float32),
                frame.Class.to_numpy(dtype=np.int64), [f'ulb:{i}' for i in frame._source_row],
                chunk_length, history_length, 'ulb:' + name, features)
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
                               prediction_targets=int(split.target_mask.sum()) if split.target_mask is not None
                               else int(split.lengths.sum()) if split.task == 'point' else len(split.X),
                               positives=int((known == 1).sum()), negatives=int((known == 0).sum()),
                               feature_names=split.feature_names)
    check_splits([load(output / f'{s}.npz') for s in ('train', 'validation', 'test')])
    report = dict(dataset=dataset, source_root=str(source_root), chunk_length=chunk_length,
                  history_length=history_length,
                  split_policy=('chronological 70/15/15 percent, timestamp ties kept together; global stream, customer IDs unavailable'
                                if dataset == 'ulb' else 'existing repository research splits; no relabelling'),
                  summaries=summaries)
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
