"""Train a wine-quality model and log everything to MLflow.

Two runs, both on the v2 data so the comparison is fair:
  baseline: logistic regression
  improved: random forest
Run:
  python ml/train.py v2 baseline
  python ml/train.py v2 improved
"""

import argparse
import subprocess
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ml" / "registry"
# the registry folder is not in git until the first run, so create it here
REGISTRY.mkdir(parents=True, exist_ok=True)
# local backend, no server needed: a SQLite database file in the project root
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_data(version):
    train = pd.read_csv(ROOT / "data" / "processed" / version / "train.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / version / "test.csv")
    y_train = train.pop("label")
    y_test = test.pop("label")
    return train, y_train, test, y_test


def build_model(name):
    if name == "baseline":
        # class_weight balanced because only ~14% of wines are "good"
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    if name == "improved":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def main(version, model_name):
    X_train, y_train, X_test, y_test = load_data(version)
    model = build_model(model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred)

    mlflow.set_experiment("wine-quality")
    run_name = f"{model_name}-{version}"
    with mlflow.start_run(run_name=run_name):
        # tags: link this run back to the code and the data
        mlflow.set_tag("git_sha", git_sha())
        mlflow.set_tag("data_version", version)
        # the DVC hash is the real fingerprint of the data (the "v2" name
        # alone is not); read it from the .dvc pointer file
        dvc_file = ROOT / "data" / "processed.dvc"
        if dvc_file.exists():
            dvc_out = yaml.safe_load(dvc_file.read_text())["outs"][0]
            mlflow.log_param("dvc_md5", dvc_out["md5"])
            mlflow.log_param("dvc_nfiles", dvc_out.get("nfiles"))
        # params
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_version", version)
        mlflow.log_param("n_train", len(X_train))
        if model_name == "improved":
            clf = model.named_steps["clf"]
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("max_depth", clf.max_depth)
        # metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        # artifacts: confusion matrix image + the model itself
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["not good", "good"]).plot(ax=ax)
        cm_path = REGISTRY / "confusion_matrix.png"
        fig.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))
        # RandomForest stores sklearn.tree._tree.Tree; we trust it because we
        # built the model ourselves right here
        mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    # the improved model is what we serve in the app
    if model_name == "improved":
        bundle = {"model": model, "features": list(X_train.columns)}
        joblib.dump(bundle, REGISTRY / "model.pkl")
        print("saved production model -> ml/registry/model.pkl")

    print(
        f"{run_name}: accuracy={acc:.3f} f1={f1:.3f} "
        f"precision={prec:.3f} recall={rec:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["v1", "v2"])
    parser.add_argument("model_name", choices=["baseline", "improved"])
    args = parser.parse_args()
    main(args.version, args.model_name)
