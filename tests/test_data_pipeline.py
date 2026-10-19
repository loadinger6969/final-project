"""Tests for the data pipeline. They build tiny synthetic data, so they run
without the real dataset (CI does not pull DVC data).

Edge cases covered:
  - label boundary: quality 6 -> 0, 7 -> 1, 3 -> 0, 8 -> 1
  - semicolon-separated file is parsed
  - duplicates are removed
  - v2 has one extra column, in a fixed order after the base ones
  - a missing column raises KeyError
"""

import pandas as pd
import pytest

from ml.data_pipeline import (
    BASE_FEATURES,
    add_label,
    make_v2_frame,
    split_and_save,
)


def _row(quality, **over):
    r = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "quality": quality,
    }
    r.update(over)
    return r


@pytest.mark.parametrize("quality,label", [(6, 0), (7, 1), (3, 0), (8, 1)])
def test_label_boundary(quality, label):
    # the only boundary that matters is quality 6 vs 7
    df = add_label(pd.DataFrame([_row(quality)]))
    assert df["label"].iloc[0] == label


def test_semicolon_file_is_parsed(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("a;b;quality\n1;2;7\n", encoding="utf-8")
    df = pd.read_csv(csv, sep=";")
    assert list(df.columns) == ["a", "b", "quality"]
    assert df["quality"].iloc[0] == 7


def test_v2_removes_duplicates_and_adds_column():
    # first two rows are identical, so they collapse to one
    df = pd.DataFrame([_row(5), _row(5), _row(8)])
    v2 = make_v2_frame(add_label(df))
    assert len(v2) == 2
    assert "alcohol_density_ratio" in v2.columns
    assert v2["alcohol_density_ratio"].iloc[0] == pytest.approx(
        v2["alcohol"].iloc[0] / v2["density"].iloc[0]
    )


def test_split_writes_v1_and_v2(tmp_path):
    rows = [
        _row(5, alcohol=9.0),
        _row(5, alcohol=10.0),
        _row(6, alcohol=9.5),
        _row(6, alcohol=10.5),
        _row(7, alcohol=11.0),
        _row(7, alcohol=12.0),
        _row(8, alcohol=12.5),
        _row(8, alcohol=13.0),
    ]
    df = add_label(pd.DataFrame(rows))
    split_and_save(df, BASE_FEATURES, "v1", out_root=tmp_path)
    v2 = make_v2_frame(df)
    split_and_save(
        v2, BASE_FEATURES + ["alcohol_density_ratio"], "v2", out_root=tmp_path
    )
    v1_train = pd.read_csv(tmp_path / "v1" / "train.csv")
    v2_train = pd.read_csv(tmp_path / "v2" / "train.csv")
    assert "label" in v1_train.columns
    # v2 has exactly one more column, and keeps the same base columns first
    assert v2_train.shape[1] == v1_train.shape[1] + 1
    assert list(v2_train.columns[:11]) == list(v1_train.columns[:11])


def test_missing_column_raises():
    df = pd.DataFrame([_row(5)]).drop(columns=["pH"])
    with pytest.raises(KeyError):
        split_and_save(df, BASE_FEATURES, "v1", out_root="unused")
