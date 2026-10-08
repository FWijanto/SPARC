"""Integrity checks for the small data artifact shipped with SPARC."""

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = REPO_ROOT / "data" / "mmp_hybrid_splits_v2_20cancer.csv"


def test_paper_split_is_complete_and_patient_disjoint():
    with SPLIT_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 44_175
    assert set(rows[0]) == {
        "slide_id",
        "fold",
        "split",
        "cancer_group",
        "patient_id",
        "cancer_type",
    }
    assert {row["split"] for row in rows} == {"train", "val"}
    assert {int(row["fold"]) for row in rows} == set(range(5))
    assert len({row["patient_id"] for row in rows}) == 7_284

    for fold in range(5):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        train = {row["patient_id"] for row in fold_rows if row["split"] == "train"}
        held_out = {row["patient_id"] for row in fold_rows if row["split"] == "val"}
        assert train
        assert held_out
        assert train.isdisjoint(held_out)
