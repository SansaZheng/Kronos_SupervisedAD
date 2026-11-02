"""Guard the actual research risks: granularity, class polarity and data leakage."""
import numpy as np
import pandas as pd
import pytest

from scripts.data.load_supervised_data import load_sapienza, load_sd_fmm, load_tstr
from scripts.data.prepare_supervised_data import DEFAULT_ROOT, FEATURES, binary_labels
from scripts.data.validate_supervised_data import validate


def test_publisher_checksums_and_prepared_alignment():
    result = validate()
    assert result == {
        "verified_source_files": 20, "verified_prepared_files": 5,
        "sapienza_rows_all_resolutions": 1887568,
        "tstr_raw_sequences": 977, "tstr_unique_sequences": 964,
        "sd_fmm_stock_sequences": 27, "sd_fmm_crypto_sequences": 337,
    }


def test_sapienza_held_out_coins_and_no_metadata_in_features():
    symbols = []
    positive = []
    for split in ("train", "validation", "test"):
        x, y, metadata = load_sapienza("15S", split)
        assert x.shape == (len(y), 12)
        assert set(metadata.columns) == {"date", "pump_index", "symbol"}
        assert not {"gt", "pump_index", "symbol", "date"} & set(FEATURES)
        assert set(np.unique(y)) == {0, 1}
        symbols.append(set(metadata.symbol))
        positive.append(y.sum())
    assert not symbols[0] & symbols[1]
    assert not symbols[0] & symbols[2]
    assert not symbols[1] & symbols[2]
    assert sum(positive) == 317
    groups = pd.read_csv(DEFAULT_ROOT / "prepared/sapienza_groups.csv")
    assert groups.groupby("symbol").split.nunique().eq(1).all()


def test_tstr_sequence_labels_and_deduplicated_partitions():
    observed = []
    for split in ("train", "validation", "test"):
        x, y, ids = load_tstr(split)
        assert x.shape == (len(y), 1000, 1) and y.ndim == 1
        assert np.isfinite(x).all() and (x > 0).all()
        np.testing.assert_array_equal(y, [str(i).startswith("new_1m_binance/") for i in ids])
        observed.extend(ids.tolist())
    assert len(observed) == len(set(observed)) == 964


@pytest.mark.parametrize("market,n,positives", [("stock", 27, 493), ("crypto", 337, 337)])
def test_sd_fmm_original_point_labels_and_disjoint_sequences(market, n, positives):
    ids, labels = [], []
    for split in ("train", "validation", "test"):
        x, y, indices = load_sd_fmm(market, split)
        assert x.shape == (len(indices), 160, 1) and y.shape == (len(indices), 160)
        ids.extend(indices.tolist())
        labels.append(y)
    assert sorted(ids) == list(range(n))
    assert np.concatenate(labels).sum() == positives


def test_unlabelled_values_cannot_be_silently_cast_to_normal():
    for bad in ([0, -1, 1], [0, np.nan, 1], [0, .5, 1]):
        with pytest.raises(ValueError):
            binary_labels(bad)


def test_split_typo_does_not_silently_return_empty_training_data():
    with pytest.raises(ValueError, match="split"):
        load_tstr("validaton")
