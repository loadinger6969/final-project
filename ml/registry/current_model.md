# Current production model

| Field | Value |
|---|---|
| File | `ml/registry/model.pkl` (4,344,346 bytes) |
| Type | scikit-learn `Pipeline`: `StandardScaler` -> `RandomForestClassifier` |
| Trees | 400, `max_depth=10`, `min_samples_leaf=3`, `class_weight="balanced"`, `random_state=42` |
| Data version | v2 (12 features, DVC md5 `5a8e5ef02766f26a2f2f9ada55364fa2.dir`) |
| Train / test rows | 1,087 / 272 |
| Accuracy | 0.846 |
| F1 (good wine) | 0.523 |
| Precision | 0.451 |
| Recall | 0.622 |
| MLflow experiment | `wine-quality` |
| MLflow run | `improved-v2`, run id `4d59d6fb1a264493b7f04ec36c46457c` |
| Trained on | 2026-10-19 10:03 |
| Commit SHA | `45c43f2d119b5d87c00a43e7a85315eda08e9669` |

## What is inside the pkl

The file is a dict with two keys:

```python
{"model": <Pipeline>, "features": [<12 column names in training order>]}
```

The feature list is saved on purpose. A model trained on a fixed column order
must receive the columns in that same order at prediction time, so the app
reorders the input using the saved list instead of trusting the order in the
request.

Feature order:

```
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol,
alcohol_density_ratio
```

## API contract

The API accepts the 11 raw measurements only. `alcohol_density_ratio` is
computed on the server as `alcohol / density`; the user never types it. This
keeps the request the same as the dataset columns and avoids the user and the
server disagreeing on the ratio.

JSON keys keep the dataset names, spaces included (`"fixed acidity"`). The
mapping from request fields to model columns lives in `ml/`, next to the model,
because it is part of the model contract and not business logic.

## Loading it in the app

```python
import joblib

bundle = joblib.load("ml/registry/model.pkl")
model, features = bundle["model"], bundle["features"]
X = frame[features]  # frame already has alcohol_density_ratio
prediction = model.predict(X)
```

`ml/registry/model.pkl` is committed to git, so the app and the Docker image do
not need DVC or the Azure remote at runtime.

## Replacing the model

1. Change the data or the hyperparameters, retrain with `python ml/train.py v2 improved`.
2. Update this file with the new metrics, run id and commit SHA.
3. Re-run `pytest tests/` so the contract tests check the new file.
