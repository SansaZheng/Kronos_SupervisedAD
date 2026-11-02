"""Verify publisher files and prepare labelled market-manipulation datasets.

Run from any directory. Requires numpy and pandas; does not download model weights.
No synthetic OHLC bars, point labels from sequence labels, or new anomaly intervals
are generated. Publisher files under raw/ are retained byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data/supervised_financial"
FEATURES = [
    "std_rush_order", "avg_rush_order", "std_trades", "std_volume",
    "avg_volume", "std_price", "avg_price", "avg_price_max",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]
SPLIT_SEED = "kronosad-supervised-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_sources(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "sources.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes dataset root")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Publisher file checksum mismatch: {item['path']}")
    return len(manifest["files"])


def group_splits(keys):
    """Deterministic 60/20/20 split of independent group IDs, never of rows."""
    keys = sorted(set(map(str, keys)), key=lambda k: hashlib.sha256(
        f"{SPLIT_SEED}:{k}".encode()).hexdigest())
    train_end, validation_end = int(len(keys) * .6), int(len(keys) * .8)
    return {key: ("train" if i < train_end else "validation" if i < validation_end
                  else "test") for i, key in enumerate(keys)}


def binary_labels(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected finite, original binary labels")
    return values.astype(np.int8)


def split_counts(labels, splits):
    return {split: {"samples": int((splits == split).sum()),
                    "positive_labels": int(labels[splits == split].sum())}
            for split in ("train", "validation", "test")}


def prepare_sapienza(root):
    tables = {resolution: pd.read_csv(root / "raw/sapienza/labeled_features" /
                                    f"features_{resolution}.csv.gz")
              for resolution in ("5S", "15S", "25S")}
    # A coin cannot occur in multiple partitions, including across resolutions.
    mapping = group_splits({symbol for df in tables.values() for symbol in df.symbol})
    manifests, summary = [], []
    for resolution, df in tables.items():
        y = binary_labels(df["gt"])
        x = df[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or df.duplicated(["pump_index", "date"]).any():
            raise ValueError(f"Invalid Sapienza feature rows: {resolution}")
        if df.groupby("pump_index").symbol.nunique().max() != 1:
            raise ValueError("A pump_index must identify a single coin")
        dates = pd.to_datetime(df.date, errors="raise", utc=True)
        if not dates.notna().all():
            raise ValueError("Missing Sapienza timestamp")
        for pump_index, group in df.groupby("pump_index", sort=True):
            symbol = group.symbol.iloc[0]
            manifests.append({
                "resolution": resolution, "pump_index": int(pump_index),
                "symbol": symbol, "split": mapping[symbol], "rows": len(group),
                "positive_labels": int(group["gt"].sum()),
                "first_date": group.date.min(), "last_date": group.date.max(),
            })
        splits = df.symbol.map(mapping).to_numpy()
        summary.append({
            "dataset": f"sapienza_{resolution}", "task": "point_onset_classification",
            "rows": len(df), "features": len(FEATURES),
            "positive_labels": int(y.sum()), "event_groups": int(df.pump_index.nunique()),
            "symbols": int(df.symbol.nunique()), "split_unit": "symbol",
            "splits": split_counts(y, splits),
        })
    pd.DataFrame(manifests).to_csv(root / "prepared/sapienza_groups.csv", index=False)
    write_json(root / "prepared/sapienza_features.json", FEATURES)
    return summary


def tstr_archive(root):
    with zipfile.ZipFile(root / "raw/tstr/v1.0.0.zip") as outer:
        names = [n for n in outer.namelist() if n.endswith("/data.zip")]
        if len(names) != 1:
            raise ValueError("Expected one publisher data.zip")
        return zipfile.ZipFile(io.BytesIO(outer.read(names[0])))


def prepare_tstr(root):
    records, unique = [], {}
    with tstr_archive(root) as archive:
        for name in sorted(archive.namelist()):
            folder = name.split("/")[0]
            if folder not in ("new_1m_binance", "random1m") or not name.endswith(".csv"):
                continue
            df = pd.read_csv(io.BytesIO(archive.read(name)))
            x = df["High"].to_numpy(dtype="<f8")
            if x.shape != (1000,) or not np.isfinite(x).all() or (x <= 0).any():
                raise ValueError(f"Invalid TSTR price series: {name}")
            label = int(folder == "new_1m_binance")
            key = hashlib.sha256(x.tobytes()).hexdigest()
            included = key not in unique
            if included:
                unique[key] = {"x": x, "label": label, "sample_id": name}
            elif unique[key]["label"] != label:
                raise ValueError("Identical TSTR series with conflicting labels")
            records.append({"sample_id": name, "label": label, "value_sha256": key,
                            "included": int(included), "canonical_id": unique[key]["sample_id"]})
    # Stratification is by sequence class; duplicate content stays in one split.
    mapping = {}
    for label in (0, 1):
        mapping.update(group_splits([k for k, v in unique.items() if v["label"] == label]))
    for record in records:
        record["split"] = mapping[record["value_sha256"]]
    pd.DataFrame(records).to_csv(root / "prepared/tstr_samples.csv", index=False)
    keys = sorted(unique)
    x = np.stack([unique[k]["x"] for k in keys])[..., None]
    y = binary_labels([unique[k]["label"] for k in keys])
    ids = np.array([unique[k]["sample_id"] for k in keys])
    splits = np.array([mapping[k] for k in keys])
    np.savez_compressed(root / "prepared/tstr_real.npz", X=x, y=y, sample_id=ids, split=splits)
    return {"dataset": "tstr_real", "task": "sequence_classification",
            "raw_sequences": len(records), "raw_positive_sequences": sum(r["label"] for r in records),
            "sequences": len(keys), "positive_labels": int(y.sum()), "length": 1000,
            "features": 1, "removed_exact_duplicates": len(records) - len(keys),
            "split_unit": "unique_price_sequence", "splits": split_counts(y, splits),
            "source_column": "High", "absolute_timestamps_available": False,
            "synthetic_series_in_default_prepared_data": False}


def prepare_sd_fmm(root):
    summary, records = [], []
    for market, directory, suffix in (("stock", "stock cases", "_bs"),
                                       ("crypto", "cryptocurrency cases", "")):
        base = root / "raw/sd_fmm" / directory
        x = np.load(base / f"test_data{suffix}.npy", allow_pickle=False)
        y = binary_labels(np.load(base / f"test_label{suffix}.npy", allow_pickle=False))
        if x.shape != (*y.shape, 1) or y.shape[1] != 160 or not np.isfinite(x).all():
            raise ValueError(f"Invalid SD-FMM {market} arrays")
        if market == "stock":
            intervals = np.load(base / "test_ano_bs.npy", allow_pickle=False)
            expected = np.zeros_like(y)
            for i, ((start, end),) in enumerate(intervals):
                expected[i, start:end + 1] = 1
            if not np.array_equal(y, expected):
                raise ValueError("SD-FMM stock intervals do not match binary labels")
        elif not (y.sum(axis=1) == 1).all():
            raise ValueError("Expected one SD-FMM crypto onset per series")
        # Publisher calls these test data; these are OUR optional research splits.
        # Hash groups prevent identical feature sequences crossing partitions.
        hashes = [hashlib.sha256(a.astype("<f8").tobytes()).hexdigest() for a in x]
        mapping = group_splits(hashes)
        for i, key in enumerate(hashes):
            records.append({"market": market, "sequence_index": i, "split": mapping[key],
                            "value_sha256": key, "positive_labels": int(y[i].sum())})
        splits = np.array([mapping[k] for k in hashes])
        summary.append({"dataset": f"sd_fmm_{market}",
                        "task": "point_interval_classification" if market == "stock"
                        else "point_onset_classification", "sequences": len(x),
                        "length": x.shape[1], "features": 1, "positive_labels": int(y.sum()),
                        "split_unit": "feature_sequence", "splits": split_counts(y, splits),
                        "absolute_timestamps_available": False,
                        "original_partition": "test", "research_splits_are_official": False})
    pd.DataFrame(records).to_csv(root / "prepared/sd_fmm_sequences.csv", index=False)
    return summary


def prepare(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    verified = verify_sources(root)
    (root / "prepared").mkdir(exist_ok=True)
    datasets = [*prepare_sapienza(root), prepare_tstr(root), *prepare_sd_fmm(root)]
    result = {"schema_version": 1, "verified_source_files": verified,
              "split_seed": SPLIT_SEED, "datasets": datasets}
    write_json(root / "summary.json", result)
    pd.json_normalize(datasets).to_csv(root / "catalog.csv", index=False)
    products = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                 "sha256": sha256(p)} for p in sorted((root / "prepared").iterdir())]
    write_json(root / "prepared_manifest.json", products)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))
