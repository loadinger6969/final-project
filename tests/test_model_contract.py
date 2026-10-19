"""Tests for the trained model contract.

The model bundle is committed to git (ml/registry/model.pkl), so these run
in CI without DVC. Edge cases:
  - bundle loads with model + features
  - features are 12 columns in a fixed order
  - predictions are in {0, 1}
  - a reversed input key order gives the same result once realigned
  - a missing feature raises KeyError
  - extreme but valid values still return a probability in [0, 1]
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "ml" / "registry" / "model.pkl"


def _valid_row():
    alcohol = 9.4
    density = 0.9978
    return {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": density,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": alcohol,
        "alcohol_density_ratio": alcohol / density,
    }


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(BUNDLE_PATH)


def test_bundle_has_model_and_features(bundle):
    assert "model" in bundle
    assert "features" in bundle
    assert len(bundle["features"]) == 12


def test_predict_returns_binary_label(bundle):
    X = pd.DataFrame([_valid_row()])[bundle["features"]]
    assert bundle["model"].predict(X)[0] in (0, 1)


def test_reversed_key_order_same_result(bundle):
    row = _valid_row()
    ordered = pd.DataFrame([row])[bundle["features"]]
    # build the same row with the keys in reverse order, then realign by name
    reversed_row = {k: row[k] for k in reversed(list(row))}
    realigned = pd.DataFrame([reversed_row])[bundle["features"]]
    assert list(realigned.columns) == list(ordered.columns)
    assert bundle["model"].predict(ordered)[0] == bundle["model"].predict(realigned)[0]


def test_missing_feature_raises(bundle):
    row = _valid_row()
    del row["pH"]
    with pytest.raises(KeyError):
        pd.DataFrame([row])[bundle["features"]]


def test_extreme_but_valid_input(bundle):
    row = _valid_row()
    row.update({"density": 0.9, "alcohol": 15.0, "pH": 4.0})
    row["alcohol_density_ratio"] = 15.0 / 0.9
    X = pd.DataFrame([row])[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[0]
    assert 0.0 <= proba[1] <= 1.0
