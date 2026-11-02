"""Use invented rows only; never include restricted competition records."""
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.data.ieee_cis_local import audit, iter_transactions, join_identity, train_labels


@pytest.fixture
def private_example(tmp_path):
    path = tmp_path / "ieee-fraud-detection.zip"
    tables = {
        "train_transaction.csv": "TransactionID,TransactionDT,isFraud\n1,10,0\n2,20,1\n",
        "train_identity.csv": "TransactionID,id_01,DeviceType\n2,3,mobile\n",
        "test_transaction.csv": "TransactionID,TransactionDT\n3,30\n4,40\n",
        "test_identity.csv": "TransactionID,id-01,DeviceType\n4,5,desktop\n",
        "sample_submission.csv": "TransactionID,isFraud\n3,0.5\n4,0.5\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in tables.items():
            archive.writestr(name, text)
    return tmp_path


def test_audit_distinguishes_submission_placeholder_from_truth(private_example):
    result = audit(private_example)
    assert result["files"]["train_transaction.csv"]["class_counts"] == {"normal": 1, "fraud": 1}
    assert result["train_transactions_without_identity"] == 1
    assert result["test_transactions_without_identity"] == 1
    assert result["test_truth_labels_available"] is False
    assert result["raw_rows_in_report"] is False


def test_left_join_preserves_transactions_and_aligns_identity_names(private_example):
    training = pd.concat(iter_transactions(private_example, "train", chunksize=1), ignore_index=True)
    testing = pd.concat(iter_transactions(private_example, "test", chunksize=1), ignore_index=True)
    assert training.TransactionID.tolist() == [1, 2]
    assert testing.TransactionID.tolist() == [3, 4]
    assert training.identity_available.tolist() == [False, True]
    assert testing.identity_available.tolist() == [False, True]
    assert "id_01" in testing and "id-01" not in testing
    assert "isFraud" not in testing
    assert np.isnan(testing.id_01.iloc[0])


def test_duplicate_identity_cannot_multiply_rows():
    with pytest.raises(ValueError, match="duplicate"):
        join_identity(pd.DataFrame({"TransactionID": [1]}),
                      pd.DataFrame({"TransactionID": [1, 1]}))


def test_fractional_submission_values_are_not_accepted_as_training_truth():
    with pytest.raises(ValueError, match="binary"):
        train_labels([0.5, 0.5])
