"""Causality, label leakage, schema safety and actual training/inference."""
import argparse
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from model import Kronos, KronosTokenizer
from supervised.data import (Split, Standardizer, check_splits, load, save,
                             validate, validate_ohlcv, prepare_builtin)
from supervised.engine import labelled_loss, predict, train
from supervised.metrics import choose_threshold, evaluate
from supervised.model import SupervisedKronos, from_specification, tiny_config

torch.set_num_threads(2)


def make_model(task='point', freeze=False):
    return SupervisedKronos(Kronos(**tiny_config()), 2, task=task, freeze_backbone=freeze)


def make_split(seed, task='point'):
    x = np.random.default_rng(seed).normal(size=(8, 6, 2)).astype(np.float32)
    y = (x[:, :, 0] > 0).astype(np.int64)
    if task == 'sequence':
        y = np.arange(8) % 2
    return Split(x, y, np.full(8, 6), np.array([f'{seed}:{i}' for i in range(8)]), ['a', 'b'])


def test_point_predictions_cannot_see_future():
    model = make_model().eval()
    x = torch.randn(2, 9, 2)
    changed = x.clone()
    changed[:, 5:] = 100 * torch.randn(2, 4, 2)
    with torch.no_grad():
        before = model(x, torch.tensor([9, 9]))
        after = model(changed, torch.tensor([9, 9]))
    torch.testing.assert_close(before[:, :5], after[:, :5])


def test_sequence_gathers_last_valid_state_and_ignores_right_padding():
    model = make_model('sequence').eval()
    x = torch.randn(2, 5, 2)
    with torch.no_grad():
        a = model(x, torch.tensor([3, 5]))
        b = model(torch.cat([x, torch.randn(2, 4, 2) * 100], dim=1), torch.tensor([3, 5]))
    torch.testing.assert_close(a, b)


def test_adapter_backpropagates_through_frozen_backbone():
    model = make_model(freeze=True).train()
    model(torch.randn(2, 5, 2), torch.tensor([5, 5])).sum().backward()
    assert model.adapter[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert not model.backbone.training


def test_unknown_labels_and_padding_do_not_enter_loss_or_metrics():
    logits = torch.tensor([0., 0., 20.], requires_grad=True)
    loss = labelled_loss(logits, torch.tensor([0., 1., -1.]), torch.tensor(1.))
    loss.backward()
    assert logits.grad[-1] == 0
    assert evaluate([0, 1, -1], [.1, .9, .99], .5)['n'] == 2
    assert labelled_loss(logits, torch.tensor([-1., -1., -1.]), torch.tensor(1.)) is None


def test_preprocessing_uses_valid_training_points_only():
    split = make_split(0)
    split.lengths[:] = 3
    split.X[:, 3:] = 10000
    split.X[0, 0, 0] = np.nan
    scaler = Standardizer().fit(split)
    np.testing.assert_allclose(scaler.mean, np.nanmean(split.X[:, :3], axis=(0, 1)), rtol=1e-6)
    saved = scaler.mean.copy()
    assert np.isfinite(scaler.transform(make_split(1).X)).all()
    np.testing.assert_array_equal(scaler.mean, saved)


def test_cross_split_content_id_and_group_overlap_rejected():
    a, b = make_split(0), make_split(1)
    check_splits([a, b])
    for kind in ('content', 'id', 'group'):
        altered = deepcopy(b)
        if kind == 'content':
            altered.X[0] = a.X[0]
        elif kind == 'id':
            altered.sample_id[0] = a.sample_id[0]
        else:
            a.group_id = np.array(['shared'] * 8)
            altered.group_id = np.array(['shared'] * 8)
        with pytest.raises(ValueError, match='overlap'):
            check_splits([a, altered])


def test_invalid_labels_and_fake_ohlc_schema_fail():
    s = make_split(0)
    s.y[0, 0] = 2
    with pytest.raises(ValueError, match='labels'):
        validate(s)
    with pytest.raises(ValueError, match='feature_names'):
        validate_ohlcv(make_split(0))
    s = make_split(0)
    s.feature_names[0] = 'is_fraud'
    with pytest.raises(ValueError, match='target/label'):
        validate(s)


def test_validation_threshold_optimizes_known_labels_only():
    assert choose_threshold([0, 1, -1], [.2, .8, .01]) == .8
    with pytest.raises(ValueError, match='both'):
        choose_threshold([0, 0], [.1, .2])


def test_sapienza_preparation_keeps_gaps_labels_and_groups(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.data.prepare_supervised_data import FEATURES
    def fake_load(resolution, split, root):
        i = {'train': 1, 'validation': 2, 'test': 3}[split]
        return (np.ones((3, len(FEATURES))) * i, np.array([0, 1, 0]),
                pd.DataFrame(dict(date=['2020-01-01 00:00:00', '2020-01-01 00:00:15',
                                         '2020-01-01 00:01:45'],
                                  pump_index=[i] * 3, symbol=[f'coin{i}'] * 3)))
    monkeypatch.setattr('scripts.data.load_supervised_data.load_sapienza', fake_load)
    prepare_builtin('sapienza', tmp_path, tmp_path / 'prepared', 4)
    s = load(tmp_path / 'prepared/train.npz')
    assert len(s.X) == 1 and s.lengths.tolist() == [3]
    assert s.y.tolist() == [[0, 1, 0, -1]]
    np.testing.assert_allclose(s.X[0, :3, -1], np.log1p([0, 15, 90]), rtol=1e-6)


def test_ulb_preparation_keeps_timestamp_ties_and_all_labels(tmp_path, monkeypatch):
    import pandas as pd
    rows = 40
    frame = pd.DataFrame({f'V{i}': np.arange(rows) + i for i in range(1, 29)})
    frame['Time'] = np.repeat(np.arange(rows // 2), 2)
    frame['Amount'] = np.arange(rows) + 1
    frame['Class'] = np.arange(rows) % 2
    monkeypatch.setattr('scripts.data.transaction_data.iter_table', lambda *a, **kw: iter([frame]))
    prepare_builtin('ulb', tmp_path, tmp_path / 'ulb', 8)
    prepared = [load(tmp_path / f'ulb/{s}.npz') for s in ('train', 'validation', 'test')]
    times = [s.X[:, :, 0][np.arange(s.X.shape[1])[None] < s.lengths[:, None]] for s in prepared]
    assert times[0].max() < times[1].min() and times[1].max() < times[2].min()
    assert sum(len(s.valid_labels()) for s in prepared) == rows
    assert sum((s.valid_labels() == 1).sum() for s in prepared) == rows // 2


def test_ohlcv_tokenizer_is_frozen_causal_and_checkpoint_roundtrips():
    config = dict(d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2,
                  n_dec_layers=2, ffn_dropout_p=0., attn_dropout_p=0.,
                  resid_dropout_p=0., s1_bits=4, s2_bits=4, beta=.05,
                  gamma0=1., gamma=1.1, zeta=.05, group_size=4)
    model = SupervisedKronos(Kronos(**tiny_config()), 6, input_kind='ohlcv',
                             tokenizer=KronosTokenizer(**config)).train()
    assert not model.tokenizer.training
    model.eval()
    x = torch.randn(2, 8, 6)
    altered = x.clone()
    altered[:, 4:] += 10
    with torch.no_grad():
        a = model(x, torch.tensor([8, 8]))
        b = model(altered, torch.tensor([8, 8]))
    torch.testing.assert_close(a[:, :4], b[:, :4])
    restored = from_specification(model.specification()).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(a, restored(x, torch.tensor([8, 8])))


@pytest.mark.parametrize('task', ['point', 'sequence'])
def test_end_to_end_test_labels_never_select_model_or_threshold(tmp_path, task):
    data = tmp_path / 'data'
    data.mkdir()
    for name, seed in zip(('train', 'validation', 'test'), (1, 2, 3)):
        save(data / f'{name}.npz', make_split(seed, task))
    args = argparse.Namespace(command='train', epochs=2, batch_size=4, lr=.001,
                              max_context=512, seed=7, threads=2, device='cpu',
                              data_dir=str(data), output=str(tmp_path / 'run1'),
                              input_kind='features', scratch=True, backbone=None,
                              tokenizer=None, freeze_backbone=False, local_only=True,
                              weight_decay=.01)
    first = train(args)
    test = load(data / 'test.npz')
    test.y = 1 - test.y
    save(data / 'test.npz', test)
    args.output = str(tmp_path / 'run2')
    second = train(args)
    assert first['best_epoch'] == second['best_epoch']
    assert first['validation'] == second['validation']
    checkpoint1 = torch.load(tmp_path / 'run1/best.pt', weights_only=True)
    checkpoint2 = torch.load(tmp_path / 'run2/best.pt', weights_only=True)
    for key in checkpoint1['state_dict']:
        torch.testing.assert_close(checkpoint1['state_dict'][key], checkpoint2['state_dict'][key])
    inference = argparse.Namespace(checkpoint=str(tmp_path / 'run1/best.pt'),
                                   data=str(data / 'test.npz'), output=str(tmp_path / 'pred.csv'),
                                   device='cpu', threads=2, batch_size=4)
    predict(inference)
    # Predicting the same features with hidden labels must not change probabilities.
    prediction = (tmp_path / 'pred.csv').read_text().splitlines()
    test.y[:] = -1
    save(data / 'test.npz', test)
    inference.output = str(tmp_path / 'unknown.csv')
    predict(inference)
    unknown = (tmp_path / 'unknown.csv').read_text().splitlines()
    assert [row.split(',')[3:] for row in prediction] == [row.split(',')[3:] for row in unknown]
    assert json.loads((tmp_path / 'run1/metrics.json').read_text())['initialization'] == 'scratch-tiny'
