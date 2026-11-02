"""Load the prepared research splits without using IDs or labels as features."""
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_supervised_data import DEFAULT_ROOT, FEATURES, binary_labels


def check_split(split):
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")


def load_sapienza(resolution="15S", split="train", root=DEFAULT_ROOT):
    """Return X(N,12), y(N), and metadata. One resolution per experiment."""
    check_split(split)
    if resolution not in ("5S", "15S", "25S"):
        raise ValueError("resolution must be 5S, 15S, or 25S")
    root = Path(root)
    groups = pd.read_csv(root / "prepared/sapienza_groups.csv")
    selected = groups[(groups.resolution == resolution) & (groups.split == split)]
    df = pd.read_csv(root / "raw/sapienza/labeled_features" / f"features_{resolution}.csv.gz")
    df = df[df.pump_index.isin(selected.pump_index)].reset_index(drop=True)
    return (df[FEATURES].to_numpy(dtype=np.float64), binary_labels(df["gt"]),
            df[["date", "pump_index", "symbol"]].copy())


def load_tstr(split="train", root=DEFAULT_ROOT):
    """Return X(N,1000,1), sequence y(N), and original sample IDs."""
    check_split(split)
    with np.load(Path(root) / "prepared/tstr_real.npz", allow_pickle=False) as archive:
        mask = archive["split"] == split
        return archive["X"][mask], archive["y"][mask], archive["sample_id"][mask]


def load_sd_fmm(market="stock", split="train", root=DEFAULT_ROOT):
    """Return X(N,160,1), point y(N,160), and source row indices.

    This optional research partition reuses publisher test files. It is not an
    official training split and must not be compared to published zero-shot scores.
    """
    check_split(split)
    if market not in ("stock", "crypto"):
        raise ValueError("market must be stock or crypto")
    root = Path(root)
    directory, suffix = ("stock cases", "_bs") if market == "stock" else ("cryptocurrency cases", "")
    base = root / "raw/sd_fmm" / directory
    manifest = pd.read_csv(root / "prepared/sd_fmm_sequences.csv")
    indices = manifest[(manifest.market == market) & (manifest.split == split)].sequence_index.to_numpy()
    x = np.load(base / f"test_data{suffix}.npy", allow_pickle=False)
    y = binary_labels(np.load(base / f"test_label{suffix}.npy", allow_pickle=False))
    return x[indices], y[indices], indices
