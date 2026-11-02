"""Check committed data integrity, label alignment and partition separation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .prepare_supervised_data import DEFAULT_ROOT, FEATURES, binary_labels, sha256, tstr_archive, verify_sources
except ImportError:
    from prepare_supervised_data import DEFAULT_ROOT, FEATURES, binary_labels, sha256, tstr_archive, verify_sources


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root=DEFAULT_ROOT):
    root = Path(root)
    verified_sources = verify_sources(root)
    products = json.loads((root / "prepared_manifest.json").read_text())
    require({p["path"] for p in products} == {
        str(p.relative_to(root)) for p in (root / "prepared").iterdir() if p.is_file()
    }, "Prepared manifest does not cover all products")
    for p in products:
        path = root / p["path"]
        require(path.stat().st_size == p["bytes"] and sha256(path) == p["sha256"],
                f"Prepared checksum mismatch: {p['path']}")
    groups = pd.read_csv(root / "prepared/sapienza_groups.csv")
    require(groups.groupby("symbol").split.nunique().max() == 1,
            "Sapienza coin leaks across partitions/resolutions")
    require(groups.groupby("pump_index").split.nunique().max() == 1,
            "Sapienza event leaks across resolutions")
    for resolution in ("5S", "15S", "25S"):
        df = pd.read_csv(root / "raw/sapienza/labeled_features" / f"features_{resolution}.csv.gz")
        y = binary_labels(df["gt"])
        require(np.isfinite(df[FEATURES].to_numpy()).all(), "Nonfinite Sapienza features")
        actual = df.groupby(["pump_index", "symbol"], as_index=False).agg(
            rows=("gt", "size"), positive_labels=("gt", "sum"))
        expected = groups[groups.resolution == resolution]
        joined = actual.merge(expected, on=["pump_index", "symbol"], how="outer", indicator=True)
        require((joined._merge == "both").all() and
                (joined.rows_x == joined.rows_y).all() and
                (joined.positive_labels_x == joined.positive_labels_y).all(),
                "Sapienza group labels/counts differ from source")
        require(y.sum() == 317, "Unexpected publisher onset labels")
    samples = pd.read_csv(root / "prepared/tstr_samples.csv")
    require(samples.groupby("value_sha256").split.nunique().max() == 1,
            "Duplicate TSTR values leak across partitions")
    canonical = samples[samples.included == 1].set_index("sample_id")
    require(canonical.value_sha256.is_unique, "TSTR prepared data is not deduplicated")
    with np.load(root / "prepared/tstr_real.npz", allow_pickle=False) as data, tstr_archive(root) as archive:
        require(data["X"].shape == (964, 1000, 1) and data["y"].shape == (964,),
                "Sequence labels must not be expanded to point labels")
        require(set(data["sample_id"]) == set(canonical.index), "Missing TSTR samples")
        for x, y, sample_id, split in zip(data["X"], data["y"], data["sample_id"], data["split"]):
            original = pd.read_csv(io.BytesIO(archive.read(str(sample_id))))["High"].to_numpy()
            require(np.array_equal(x[:, 0], original), "TSTR prices differ from publisher")
            require(int(y) == int(str(sample_id).startswith("new_1m_binance/")), "TSTR class reversal")
            require(split == canonical.loc[sample_id, "split"], "TSTR split alignment error")
    sequences = pd.read_csv(root / "prepared/sd_fmm_sequences.csv")
    for market, directory, suffix in (("stock", "stock cases", "_bs"),
                                       ("crypto", "cryptocurrency cases", "")):
        base = root / "raw/sd_fmm" / directory
        x = np.load(base / f"test_data{suffix}.npy", allow_pickle=False)
        y = binary_labels(np.load(base / f"test_label{suffix}.npy", allow_pickle=False))
        records = sequences[sequences.market == market].sort_values("sequence_index")
        require(records.sequence_index.tolist() == list(range(len(x))), "Missing SD-FMM sequences")
        require(x.shape == (*y.shape, 1), "SD-FMM features and point labels do not align")
        hashes = [hashlib.sha256(a.astype("<f8").tobytes()).hexdigest() for a in x]
        require(hashes == records.value_sha256.tolist(), "SD-FMM sequence order mismatch")
        require(records.groupby("value_sha256").split.nunique().max() == 1,
                "SD-FMM duplicate feature sequence leaks across partitions")
        require(np.array_equal(y.sum(axis=1), records.positive_labels), "SD-FMM label count mismatch")
    return {"verified_source_files": verified_sources, "verified_prepared_files": len(products),
            "sapienza_rows_all_resolutions": int(groups.rows.sum()),
            "tstr_raw_sequences": len(samples), "tstr_unique_sequences": len(canonical),
            "sd_fmm_stock_sequences": int((sequences.market == "stock").sum()),
            "sd_fmm_crypto_sequences": int((sequences.market == "crypto").sum())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    print(json.dumps(validate(parser.parse_args().root), indent=2))
