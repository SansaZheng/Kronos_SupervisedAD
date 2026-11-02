"""Verify, stream, and restore the complete payment/AML data archives.

Files are byte-split into <=40 MiB parts for GitHub. No rows or labels are
removed. ZIP/TGZ files remain original; CSVs from HF mirrors are losslessly
gzipped. This module does not download weights, fit models, or create splits.
"""
from __future__ import annotations

import argparse
import bisect
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data/supervised_financial/transactions"


class PartReader(io.RawIOBase):
    """A bounded-memory, seekable view over byte-concatenated files."""

    def __init__(self, paths):
        super().__init__()
        self.paths = list(map(Path, paths))
        self.starts = [0]
        for path in self.paths:
            self.starts.append(self.starts[-1] + path.stat().st_size)
        self.position = 0
        self.handle = None
        self.active = None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=0):
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError("Invalid whence")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.starts[-1])
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def readinto(self, buffer):
        self._checkClosed()
        view = memoryview(buffer).cast("B")
        written = 0
        while written < len(view) and self.position < self.starts[-1]:
            index = bisect.bisect_right(self.starts, self.position) - 1
            if self.active != index:
                if self.handle is not None:
                    self.handle.close()
                self.handle = self.paths[index].open("rb")
                self.active = index
            self.handle.seek(self.position - self.starts[index])
            count = min(len(view) - written, self.starts[index + 1] - self.position)
            n = self.handle.readinto(view[written:written + count])
            if not n:
                raise OSError("A data part changed while reading")
            written += n
            self.position += n
        return written

    def close(self):
        if self.handle is not None:
            self.handle.close()
        super().close()


def manifest(root=ROOT):
    return json.loads((Path(root) / "assets.json").read_text())


def asset_info(asset_id, root=ROOT):
    for asset in manifest(root)["assets"]:
        if asset["id"] == asset_id:
            return asset
    raise ValueError(f"Unknown public asset {asset_id!r}; restricted IEEE-CIS uses ieee_cis_local.py")


def part_paths(asset, root):
    root = Path(root).resolve()
    paths = [(root / part["path"]).resolve() for part in asset["parts"]]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Part path escapes data root")
    return paths


def open_asset(asset_id, root=ROOT):
    asset = asset_info(asset_id, root)
    return io.BufferedReader(PartReader(part_paths(asset, root)))


def stream_digest(stream):
    digest, size = hashlib.sha256(), 0
    while chunk := stream.read(2**20):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def verify(root=ROOT, source_bytes=False):
    """Check each part and reconstructed archive; optionally check original CSVs."""
    root = Path(root)
    part_count, total = 0, 0
    for asset in manifest(root)["assets"]:
        paths = part_paths(asset, root)
        for path, part in zip(paths, asset["parts"]):
            with path.open("rb") as stream:
                size, digest = stream_digest(stream)
            if size != part["bytes"] or digest != part["sha256"]:
                raise ValueError(f"Corrupt part: {part['path']}")
            part_count += 1
        with open_asset(asset["id"], root) as stream:
            size, digest = stream_digest(stream)
        if size != asset["bytes"] or digest != asset["sha256"]:
            raise ValueError(f"Reconstructed archive mismatch: {asset['id']}")
        if asset.get("publisher_lfs_sha256") and digest != asset["publisher_lfs_sha256"]:
            raise ValueError("Archive differs from IBM publisher's Git LFS hash")
        if source_bytes and asset.get("source_sha256"):
            with open_asset(asset["id"], root) as stream, gzip.GzipFile(fileobj=stream) as decoded:
                size, digest = stream_digest(decoded)
            if size != asset["source_bytes"] or digest != asset["source_sha256"]:
                raise ValueError(f"Decompressed bytes differ from pinned mirror: {asset['id']}")
        total += asset["bytes"]
    for notice in json.loads((root / "notices_manifest.json").read_text()):
        with (root / notice["path"]).open("rb") as stream:
            size, digest = stream_digest(stream)
        if size != notice["bytes"] or digest != notice["sha256"]:
            raise ValueError(f"Notice changed: {notice['path']}")
    return {"assets": len(manifest(root)["assets"]), "parts": part_count,
            "archive_bytes": total, "decompressed_source_hashes_checked": source_bytes}


@contextmanager
def csv_stream(asset_id, root=ROOT):
    asset = asset_info(asset_id, root)
    with open_asset(asset_id, root) as stream:
        if asset["format"] == "zip":
            with zipfile.ZipFile(stream) as archive, archive.open(asset["member"]) as csv:
                yield csv
        elif asset["format"] == "tar.gz":
            # Seekable mode also supports pandas TextIOWrapper.seekable().
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                for member in archive:
                    if member.name == asset["member"]:
                        with archive.extractfile(member) as csv:
                            yield csv
                        return
                raise ValueError("Publisher CSV missing from TGZ")
        elif asset["format"] == "gzip":
            with gzip.GzipFile(fileobj=stream) as csv:
                yield csv
        else:
            raise ValueError("Unsupported archive format")


def iter_table(asset_id, chunksize=100_000, root=ROOT, **read_csv_options):
    """Yield native CSV rows, keeping all original columns and label values.

    Elliptic features have no header. IBM's duplicate Account headers become
    Account and Account.1 in pandas; account identifiers should remain strings.
    Use nrows/usecols/dtype as needed. Pattern text is not a feature table.
    """
    asset = asset_info(asset_id, root)
    if asset_id == "ibm_aml_patterns":
        raise ValueError("Pattern annotations are not a CSV feature table")
    if "csv_header" in asset:
        read_csv_options.setdefault("header", asset["csv_header"])
    with csv_stream(asset_id, root) as stream:
        yield from pd.read_csv(stream, chunksize=chunksize, **read_csv_options)


def map_labels(dataset, values):
    """Return 1=anomaly, 0=normal, -1=unknown; unknown is never a negative."""
    mapping = {"tabformer": {"Yes": 1, "No": 0},
               "elliptic": {"1": 1, "2": 0, "unknown": -1},
               "ulb": {"0": 0, "1": 1}, "paysim": {"0": 0, "1": 1},
               "ibm_aml": {"0": 0, "1": 1}}
    if dataset not in mapping:
        raise ValueError("Unknown dataset label semantics")
    values = pd.Series(values).astype(str).str.strip()
    converted = values.map(mapping[dataset])
    if converted.isna().any():
        raise ValueError(f"Unexpected {dataset} label value")
    return converted.to_numpy(dtype=np.int8)


def restore(asset_id, output_dir, root=ROOT):
    """Reassemble one exact archive outside Git; never overwrite an existing file."""
    asset = asset_info(asset_id, root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / asset["filename"]
    with destination.open("xb") as target, open_asset(asset_id, root) as source:
        shutil.copyfileobj(source, target, length=2**20)
    with destination.open("rb") as source:
        size, digest = stream_digest(source)
    if size != asset["bytes"] or digest != asset["sha256"]:
        raise ValueError(f"Restored archive failed checksum: {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "restore"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-bytes", action="store_true", help="also verify decompressed HF CSV bytes")
    parser.add_argument("--asset", help="asset ID from assets.json")
    parser.add_argument("--output", type=Path, help="directory for the reconstructed archive")
    args = parser.parse_args()
    if args.action == "verify":
        print(json.dumps(verify(args.root, args.source_bytes), indent=2))
    else:
        if not args.asset or not args.output:
            parser.error("restore requires --asset and --output")
        print(restore(args.asset, args.output, args.root))
