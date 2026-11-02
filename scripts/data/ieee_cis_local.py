"""Audit and stream a user's locally authorized IEEE-CIS competition download.

No authentication is exported, no rules are accepted automatically, and no raw
data is uploaded. Test labels remain absent; sample_submission is not truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

EXPECTED = {
    "train_transaction.csv", "train_identity.csv", "test_transaction.csv",
    "test_identity.csv", "sample_submission.csv",
}


def members(archive):
    entries = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name not in EXPECTED or name in entries:
            raise ValueError("Expected exactly the five official CSV files")
        entries[name] = info.filename
    if set(entries) != EXPECTED:
        raise ValueError("Missing official IEEE-CIS CSV files")
    return entries


def archive_path(directory):
    path = Path(directory)
    return path if path.is_file() else path / "ieee-fraud-detection.zip"


def read_table(archive, entries, name, **options):
    with archive.open(entries[name]) as stream:
        return pd.read_csv(stream, **options)


def require_ids(frame, description):
    if frame.TransactionID.isna().any() or not frame.TransactionID.is_unique:
        raise ValueError(f"Missing or duplicate TransactionID in {description}")


def train_labels(values):
    values = np.asarray(values)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Training labels must be original binary isFraud values")
    return values.astype(np.int8)


def audit(directory, check_crc=True):
    path = archive_path(directory)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2**20):
            digest.update(chunk)
    result = {
        "dataset": "IEEE-CIS Fraud Detection", "storage": "local_only",
        "source_url": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        "rules_url": "https://www.kaggle.com/competitions/ieee-fraud-detection/rules",
        "archive_bytes": path.stat().st_size, "archive_sha256": digest.hexdigest(),
        "crc_checked": check_crc, "files": {},
    }
    with zipfile.ZipFile(path) as archive:
        entries = members(archive)
        if check_crc and archive.testzip() is not None:
            raise ValueError("Downloaded ZIP failed CRC validation")
        tables = {}
        for name in sorted(EXPECTED):
            header = read_table(archive, entries, name, nrows=0).columns.tolist()
            if "TransactionID" not in header:
                raise ValueError("Missing transaction key")
            if name == "train_transaction.csv":
                required = ["TransactionID", "TransactionDT", "isFraud"]
            elif name == "test_transaction.csv":
                if "isFraud" in header:
                    raise ValueError("Official test transactions must not contain truth labels")
                required = ["TransactionID", "TransactionDT"]
            else:
                required = ["TransactionID"]
            frame = read_table(archive, entries, name, usecols=required)
            require_ids(frame, name)
            tables[name] = frame
            info = archive.getinfo(entries[name])
            summary = {"rows": len(frame), "columns": len(header),
                       "uncompressed_bytes": info.file_size, "csv_header": header}
            if name == "train_transaction.csv":
                y = train_labels(frame.isFraud)
                summary["class_counts"] = {"normal": int((y == 0).sum()),
                                            "fraud": int((y == 1).sum())}
            if "TransactionDT" in frame:
                summary["relative_time_min"] = int(frame.TransactionDT.min())
                summary["relative_time_max"] = int(frame.TransactionDT.max())
            result["files"][name] = summary
        train_ids = set(tables["train_transaction.csv"].TransactionID)
        test_ids = set(tables["test_transaction.csv"].TransactionID)
        if train_ids & test_ids:
            raise ValueError("Official train and test transaction IDs overlap")
        for split, ids in (("train", train_ids), ("test", test_ids)):
            identity = tables[f"{split}_identity.csv"]
            if not set(identity.TransactionID) <= ids:
                raise ValueError("Identity key does not belong to its transaction split")
            result[f"{split}_transactions_with_identity"] = len(identity)
            result[f"{split}_transactions_without_identity"] = len(ids) - len(identity)
        if not np.array_equal(tables["sample_submission.csv"].TransactionID,
                              tables["test_transaction.csv"].TransactionID):
            raise ValueError("Sample submission does not match test transaction order")
    result["test_truth_labels_available"] = False
    result["raw_rows_in_report"] = False
    return result


def normalize_identity_columns(frame):
    """Official test uses id-XX where training uses id_XX."""
    mapping = {column: re.sub(r"^id-(\d+)$", r"id_\1", column)
               for column in frame.columns}
    renamed = frame.rename(columns=mapping)
    if not renamed.columns.is_unique:
        raise ValueError("Identity column normalization would create duplicates")
    return renamed


def join_identity(transactions, identity):
    require_ids(transactions, "transaction chunk")
    require_ids(identity, "identity table")
    if "identity_available" in transactions or "identity_available" in identity:
        raise ValueError("Reserved identity availability field already exists")
    identity = normalize_identity_columns(identity)
    result = transactions.merge(identity, on="TransactionID", how="left",
                                validate="one_to_one", sort=False, indicator=True)
    result["identity_available"] = result.pop("_merge").eq("both")
    return result


def iter_transactions(directory, split="train", chunksize=50_000, with_identity=True):
    """Yield transaction chunks. Only train contains isFraud; test stays unlabelled.

    Run audit first. Identity rows missing from the source stay missing after a
    left join; no transaction is dropped. CSV row values are never logged.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    with zipfile.ZipFile(archive_path(directory)) as archive:
        entries = members(archive)
        identity = read_table(archive, entries, f"{split}_identity.csv") if with_identity else None
        with archive.open(entries[f"{split}_transaction.csv"]) as stream:
            for chunk in pd.read_csv(stream, chunksize=chunksize, low_memory=False):
                if split == "train":
                    train_labels(chunk["isFraud"])
                elif "isFraud" in chunk:
                    raise ValueError("Official test must remain unlabelled")
                yield join_identity(chunk, identity) if with_identity else chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="local private directory or ZIP file")
    parser.add_argument("--report", type=Path, help="save aggregate audit report locally")
    args = parser.parse_args()
    result = audit(args.directory)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
        args.report.chmod(0o600)
    # Only aggregate counts; no row values, participant identity or login URL.
    print(json.dumps({"archive_bytes": result["archive_bytes"],
                      "crc_checked": result["crc_checked"],
                      "files": {name: {k: v for k, v in record.items() if k != "csv_header"}
                                for name, record in result["files"].items()},
                      "test_truth_labels_available": False}, indent=2))
