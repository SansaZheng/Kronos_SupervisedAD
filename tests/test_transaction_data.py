"""Data integrity, archive boundaries, and original fraud-label semantics."""
import io
import json

import numpy as np
import pytest

from scripts.data.transaction_data import (
    ROOT, PartReader, iter_table, manifest, map_labels, verify,
)


def test_multipart_seek_and_reads_cross_boundaries(tmp_path):
    paths = []
    for i, value in enumerate([b"abc", b"defgh", b"ijk"]):
        path = tmp_path / str(i)
        path.write_bytes(value)
        paths.append(path)
    with io.BufferedReader(PartReader(paths), buffer_size=2) as stream:
        assert stream.read(5) == b"abcde"
        assert stream.seek(-3, 2) == 8
        assert stream.read() == b"ijk"
        stream.seek(2)
        assert stream.read(7) == b"cdefghi"
        stream.seek(30)
        assert stream.read(1) == b""
        with pytest.raises(ValueError):
            stream.seek(-1)


def test_all_downloaded_parts_and_original_archive_hashes():
    result = verify()
    assert result["assets"] == 9
    assert result["archive_bytes"] > 700_000_000
    assert all(part["bytes"] <= 40 * 2**20 for asset in manifest()["assets"]
               for part in asset["parts"])


def test_elliptic_unknown_is_not_normal_and_class_one_is_illicit():
    np.testing.assert_array_equal(map_labels("elliptic", ["1", "2", "unknown"]), [1, 0, -1])
    np.testing.assert_array_equal(map_labels("tabformer", ["Yes", "No"]), [1, 0])
    with pytest.raises(ValueError):
        map_labels("elliptic", [None])
    with pytest.raises(ValueError):
        map_labels("ulb", [0, -1, 1])


@pytest.mark.parametrize("asset,target", [
    ("ulb", "Class"), ("paysim", "isFraud"),
    ("tabformer", "Is Fraud?"), ("ibm_aml", "Is Laundering"),
    ("elliptic_classes", "class"),
])
def test_original_csv_headers_and_label_columns_survive_byte_splitting(asset, target):
    frame = next(iter_table(asset, nrows=50))
    assert len(frame) == 50 and target in frame
    dataset = "elliptic" if asset == "elliptic_classes" else asset
    assert set(map_labels(dataset, frame[target])) <= {-1, 0, 1}


def test_complete_public_file_counts_and_local_only_competition_are_explicit():
    catalog = json.loads((ROOT / "catalog.json").read_text())
    downloaded = {d["id"]: d for d in catalog if d["status"] == "downloaded"}
    assert set(downloaded) == {"ulb", "elliptic", "paysim", "tabformer", "ibm_aml"}
    assert downloaded["paysim"]["rows"] == 6362620
    assert downloaded["tabformer"]["positive_labels"] == 29757
    assert downloaded["ibm_aml"]["positive_labels"] == 5177
    restricted = next(d for d in catalog if d["id"] == "ieee_cis")
    assert restricted["status"] == "local_only_restricted"
    assert restricted["rows"] == 590540 and restricted["positive_labels"] == 20663
    assert restricted["raw_data_in_repository"] is False
