# Experiments

Both runs use dataset v2 (`5a8e5ef02766f26a2f2f9ada55364fa2.dir`), the same
test split and the same random seed, so only the model changes.

## baseline (logistic regression)

| metric | value |
|---|---|
| accuracy | 0.790 |
| F1 | 0.513 |
| precision | 0.375 |
| recall | 0.811 |

Exact values: accuracy 0.7904, F1 0.5128, precision 0.3750, recall 0.8108.

Settings: `StandardScaler` then `LogisticRegression(max_iter=1000,
class_weight="balanced")`. 1,087 training rows.

## improved (random forest)

| metric | value |
|---|---|
| accuracy | 0.846 |
| F1 | 0.523 |
| precision | 0.451 |
| recall | 0.622 |

Exact values: accuracy 0.8456, F1 0.5227, precision 0.4510, recall 0.6216.

Settings: `StandardScaler` then `RandomForestClassifier(n_estimators=400,
max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42)`.

## Which run we put in production, and why

The random forest. Accuracy, precision and F1 are all higher, and the gain in
precision is the one that matters for the app: 0.375 -> 0.451 means that when
the app says "good wine", it is right more often.

The cost is recall: 0.811 -> 0.622. The logistic regression catches more of the
good wines, but it also calls a lot of ordinary wines "good", so the user gets
more false alarms. We preferred fewer false alarms over catching every good
wine.

The F1 gain is small (0.013) and we do not present it as a big improvement.
The honest summary is that the random forest trades recall for precision.

## The metric we optimise, and why

F1 of the positive class ("good wine"), not accuracy.

Only 13.6% of the wines are good (217 of 1,599). A model that always answers
"not good" would already score 0.864 accuracy while being useless. Accuracy
therefore does not tell us whether the model learned anything, and F1 balances
precision and recall on the class we actually care about.

For the app, the practical question is: when the user gets a "good" answer, is
it reliable? That is precision. F1 is our main number because it keeps the
recall side visible instead of optimising precision alone.

## Notes

- There are four runs in MLflow. One `baseline-v2` is marked FAILED: it crashed
  while writing `confusion_matrix.png` because `ml/registry/` did not exist yet
  on the first run. The code now creates the folder, and the next run finished.
- `improved-v2` was run twice (18 Oct 20:56 and 19 Oct 10:03). Both runs give
  exactly the same metrics, because the seed is fixed and the data is the same.
  The model in `ml/registry/model.pkl` comes from the second, later run.
- Every run logs `git_sha` (`45c43f2...`) and `dvc_md5`
  (`5a8e5ef0...`), so a run can be traced back to one code commit and one data
  version.
