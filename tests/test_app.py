"""Tests for the wine prediction API.

Three meaningful groups, each with several edge cases:
  1. health / sample endpoints
  2. predictions on known wines
  3. invalid inputs are rejected with 422
Run: pytest -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# real rows from the dataset: this one is quality 5, this one is quality 8
BAD_WINE = {
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
}
GOOD_WINE = {
    "fixed acidity": 12.6,
    "volatile acidity": 0.31,
    "citric acid": 0.72,
    "residual sugar": 2.2,
    "chlorides": 0.072,
    "free sulfur dioxide": 6.0,
    "total sulfur dioxide": 15.0,
    "density": 0.9968,
    "pH": 3.23,
    "sulphates": 0.82,
    "alcohol": 12.8,
}


# ---------- group 1: health and sample ----------
def test_root_is_healthy():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sample_feedbacks_into_predict():
    # the sample endpoint must return an input the API accepts
    sample = client.get("/sample").json()
    r = client.post("/predict", json=sample)
    assert r.status_code == 200


# ---------- group 2: predictions ----------
def test_known_bad_wine_is_not_good():
    r = client.post("/predict", json=BAD_WINE)
    assert r.status_code == 200
    assert r.json()["prediction"] == "not good"


def test_known_good_wine_is_good():
    r = client.post("/predict", json=GOOD_WINE)
    assert r.json()["prediction"] == "good"


@pytest.mark.parametrize("wine", [BAD_WINE, GOOD_WINE])
def test_probability_is_between_0_and_1(wine):
    p = client.post("/predict", json=wine).json()["good_probability"]
    assert 0.0 <= p <= 1.0


# ---------- group 3: invalid inputs ----------
@pytest.mark.parametrize("missing", ["pH", "alcohol", "chlorides", "density"])
def test_missing_field_is_rejected(missing):
    bad = dict(BAD_WINE)
    bad.pop(missing)
    assert client.post("/predict", json=bad).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("pH", 9.0),
        ("alcohol", 50.0),
        ("density", 5.0),
        ("fixed acidity", 200.0),
    ],
)
def test_value_out_of_range_is_rejected(field, value):
    bad = dict(BAD_WINE)
    bad[field] = value
    assert client.post("/predict", json=bad).status_code == 422


def test_wrong_type_is_rejected():
    bad = dict(BAD_WINE)
    bad["alcohol"] = "lots"
    assert client.post("/predict", json=bad).status_code == 422
