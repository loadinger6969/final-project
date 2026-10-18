"""Turn the raw wine csv into trainable datasets (v1 and v2).

v1: raw data as-is, only split into train/test.
v2: duplicates removed + one extra feature (alcohol / density).
Run: python ml/data_pipeline.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "winequality-red.csv"
PROC = ROOT / "data" / "processed"
RANDOM_STATE = 42

BASE_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def add_label(df):
    """Add the binary label: a wine is good when quality is 7 or 8."""
    df = df.copy()
    df["label"] = (df["quality"] >= 7).astype(int)
    return df


def load_raw():
    # the raw file is separated by semicolons, not commas
    df = pd.read_csv(RAW_PATH, sep=";")
    return add_label(df)


def make_v2_frame(df):
    """v2 cleaning: remove duplicates, then add alcohol / density."""
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"removed duplicates: {before - len(df)}")
    df["alcohol_density_ratio"] = df["alcohol"] / df["density"]
    return df


def split_and_save(df, features, version, out_root=PROC):
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=True)
    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    train = X_train.copy()
    train["label"] = y_train
    test = X_test.copy()
    test["label"] = y_test
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"{version}: train {len(train)}, test {len(test)}, features {len(features)}")


def make_v1():
    df = load_raw()
    split_and_save(df, BASE_FEATURES, "v1")


def make_v2():
    df = make_v2_frame(load_raw())
    features = BASE_FEATURES + ["alcohol_density_ratio"]
    split_and_save(df, features, "v2")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version", choices=["v1", "v2", "all"], nargs="?", default="all"
    )
    args = parser.parse_args()
    if args.version in ("v1", "all"):
        make_v1()
    if args.version in ("v2", "all"):
        make_v2()
