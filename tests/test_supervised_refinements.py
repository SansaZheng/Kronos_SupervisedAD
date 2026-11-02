"""Temporal ownership, causal heads, budget constraints, and protected holdouts."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from model import Kronos
from supervised.__main__ import parser
from supervised.data import (Standardizer, check_splits, chunk_point_stream, load,
                             save, validate)
from supervised.engine import (loader, score, train, write_predictions,
                               isolation_reference, check_evaluation_isolation)
from supervised.metrics import evaluate, select_operating_point
from supervised.model import SupervisedKronos, causal_mean, from_specification, tiny_config
from supervised.optimization import (labelled_loss, make_optimizer, make_scheduler,
                                     positive_class_weight)


def stream(prefix='train'):
    x = np.arange(34, dtype=np.float32).reshape(17, 2) / 10
    y = np.arange(17) % 2
    return chunk_point_stream(x, y, [f'{prefix}:{i}' for i in range(17)],
                              4, 3, prefix, ['a', 'b'], group=prefix)


def test_history_reuses_only_past_and_owns_every_target_once(tmp_path):
    s = stream()
    assert s.lengths.tolist() == [4, 7, 7, 7, 4]
    assert s.target_mask.sum() == 17
    assert len(s.valid_labels()) == 17
    np.testing.assert_array_equal(s.X[1, :3], s.X[0, 1:4])
    assert not s.target_mask[1, :3].any()
    assert np.all(s.y[1, :3] == -1)
    scaler = Standardizer().fit(s)
    np.testing.assert_allclose(scaler.mean, np.arange(34).reshape(17, 2).mean(0) / 10, rtol=1e-6)
    model = SupervisedKronos(Kronos(**tiny_config()), 2).eval()
    scored = score(model, loader(s, scaler, 2), torch.device('cpu'))
    assert len(scored[0]) == 17
    write_predictions(tmp_path / 'pred.csv', s, scored, .5)
    import pandas as pd
    table = pd.read_csv(tmp_path / 'pred.csv')
    assert table.observation_id.is_unique
    assert table.observation_id.tolist() == [f'train:{i}' for i in range(17)]
    save(tmp_path / 's.npz', s)
    np.testing.assert_array_equal(load(tmp_path / 's.npz').target_mask, s.target_mask)


def test_context_cannot_overlap_another_split_even_with_distinct_targets():
    a, b = stream('a'), stream('b')
    check_splits([a, b])
    b.observation_id[1, 0] = a.observation_id[0, 0]
    with pytest.raises(ValueError, match='overlap'):
        check_splits([a, b])
    bad = deepcopy(a)
    bad.observation_id[1, 3] = bad.observation_id[0, 0]
    with pytest.raises(ValueError, match='only once'):
        validate(bad)


def test_frozen_checkpoint_evaluation_rejects_training_sources():
    reference = isolation_reference([stream('train'), stream('validation')])
    assert check_evaluation_isolation(stream('test'), reference)
    with pytest.raises(ValueError, match='overlaps'):
        check_evaluation_isolation(stream('train'), reference)
    hidden = stream('test')
    hidden.observation_id[0, 0] = 'train:0'
    with pytest.raises(ValueError, match='overlaps'):
        check_evaluation_isolation(hidden, reference)
    hidden.observation_id = None
    with pytest.raises(ValueError, match='needs observation_id'):
        check_evaluation_isolation(hidden, reference)
    assert not check_evaluation_isolation(stream('test'), None)


def test_target_mask_cannot_silently_label_padding():
    s = stream()
    s.target_mask[0, -1] = True
    with pytest.raises(ValueError, match='valid observations'):
        validate(s)


def test_unsupervised_only_chunks_do_not_consume_optimizer_schedule():
    s = stream()
    s.y[1:] = -1
    batches = loader(s, Standardizer().fit(s), batch_size=2, shuffle=True)
    assert len(batches) == 1
    emitted = [int(i) for _, _, _, indices in batches for i in indices]
    assert emitted == [0]
    assert len(loader(s, Standardizer().fit(s), batch_size=2)) == 3


def test_multiscale_mean_and_point_head_are_causal():
    h = torch.tensor([[[1.], [3.], [8.], [4.]]])
    torch.testing.assert_close(causal_mean(h, 2), torch.tensor([[[1.], [2.], [5.5], [6.]]]))
    model = SupervisedKronos(Kronos(**tiny_config()), 2, head_type='multiscale').eval()
    x = torch.randn(2, 12, 2)
    changed = x.clone()
    changed[:, 7:] += 100
    with torch.no_grad():
        a = model(x, torch.tensor([12, 12]))
        b = model(changed, torch.tensor([12, 12]))
    torch.testing.assert_close(a[:, :7], b[:, :7])
    restored = from_specification(model.specification()).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(a, restored(x, torch.tensor([12, 12])))


def test_boosting_control_features_use_no_future_observations():
    from supervised.baseline import temporal_features, design
    x = np.random.default_rng(1).normal(size=(2, 16, 2)).astype(np.float32)
    altered = x.copy()
    altered[:, 8:] += 100
    np.testing.assert_allclose(temporal_features(x)[:, :8], temporal_features(altered)[:, :8], rtol=1e-6)
    s = stream()
    features, labels, _, _ = design(s, Standardizer().fit(s))
    assert len(features) == len(labels) == 17


def test_attention_pool_ignores_padding_and_has_trainable_weights():
    model = SupervisedKronos(Kronos(**tiny_config()), 2, task='sequence',
                             sequence_pooling='attention').eval()
    x = torch.randn(2, 5, 2)
    with torch.no_grad():
        a = model(x, torch.tensor([3, 5]))
        b = model(torch.cat([x, torch.randn(2, 4, 2) * 100], 1), torch.tensor([3, 5]))
    torch.testing.assert_close(a, b)
    model(x, torch.tensor([3, 5])).sum().backward()
    assert model.pool_score.weight.grad.abs().sum() > 0


def test_legacy_model_specifications_still_load():
    model = SupervisedKronos(Kronos(**tiny_config()), 2)
    spec = model.specification()
    for field in ('head_type', 'sequence_pooling', 'scales', 'head_dropout'):
        spec.pop(field)
    restored = from_specification(spec)
    restored.load_state_dict(model.state_dict(), strict=True)


def test_focal_reduces_easy_example_influence_and_masks_unknowns():
    y = torch.tensor([1., 0., -1.])
    logits = torch.tensor([5., -5., 100.], requires_grad=True)
    weight = torch.tensor(2.)
    bce = labelled_loss(logits, y, weight)
    torch.testing.assert_close(bce, labelled_loss(logits, y, weight, 'focal', 0))
    focal = labelled_loss(logits, y, weight, 'focal', 2)
    assert 0 < focal < bce
    focal.backward()
    assert logits.grad[-1] == 0 and torch.isfinite(logits.grad).all()
    assert positive_class_weight(np.array([0] * 100 + [1]), 'sqrt') == 10
    assert positive_class_weight(np.array([0] * 100 + [1]), 'auto', 20) == 20


def test_optimizer_covers_trainable_parameters_once_and_preserves_lr_ratio():
    model = SupervisedKronos(Kronos(**tiny_config()), 2, head_type='multiscale')
    optimizer, groups = make_optimizer(model, .001, .01, .1, .8)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert max(g['lr'] for g in groups if g['role'] == 'backbone') == .0001
    assert all(g['lr'] == .001 for g in groups if g['role'] == 'head_adapter')
    scheduler = make_scheduler(optimizer, 10, 'cosine', .2)
    initial = [g['lr'] for g in optimizer.param_groups]
    optimizer.step(); scheduler.step()
    assert all(g['lr'] > old for g, old in zip(optimizer.param_groups, initial))
    for _ in range(9):
        optimizer.step(); scheduler.step()
    assert all(g['lr'] == 0 for g in optimizer.param_groups)


def test_budget_thresholds_keep_ties_whole_and_allow_no_alerts():
    y, p = [1, 0, 1, 0], [.9, .8, .8, .1]
    selected = select_operating_point(y, p, policy='fpr', max_fpr=0)
    assert selected['validation_fpr'] == 0 and selected['threshold'] == .9
    assert evaluate(y, p, selected['threshold'])['fp'] == 0
    selected = select_operating_point([0, 1], [.9, .8], policy='fpr', max_fpr=0)
    assert selected['validation_alerts'] == 0
    assert evaluate([0, 1], [.9, .8], selected['threshold'])['tp'] == 0
    impossible = select_operating_point([0, 1], [.9, .8], policy='precision', min_precision=.9)
    assert not impossible['feasible'] and impossible['validation_alerts'] == 0


@pytest.mark.parametrize('scores', [[np.nan, .5], [-.1, .5], [.5, 1.1]])
def test_invalid_probability_scores_fail(scores):
    with pytest.raises(ValueError, match='finite'):
        select_operating_point([0, 1], scores)


def test_comparison_does_not_invent_statistics_for_undefined_metrics():
    from scripts.compare_supervised_heads import summarize_metric
    assert summarize_metric([None, None]) == dict(mean=None, std=None, n=0)
    assert summarize_metric([.4, None]) == dict(mean=.4, std=None, n=1)


def test_validation_only_early_stopping_never_opens_test(tmp_path, monkeypatch):
    import supervised.engine as engine
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('train', 'validation'):
        save(data / f'{name}.npz', stream(name))
    args = parser().parse_args(['train', '--scratch', '--data-dir', str(data),
        '--output', str(tmp_path / 'run'), '--epochs', '20', '--defer-test',
        '--patience', '1', '--min-delta', '2', '--loss', 'focal', '--head-dropout', '0'])
    original_load = engine.load
    def guarded_load(path):
        assert path.name != 'test.npz'
        return original_load(path)
    monkeypatch.setattr(engine, 'load', guarded_load)
    report = train(args)
    assert report['best_epoch'] == 1 and report['epochs_executed'] == 2
    assert report['test'] is None and report['test_deferred']
    assert not (tmp_path / 'run/test_predictions.csv').exists()
