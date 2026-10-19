# Data provenance

## Where the data comes from

UCI Machine Learning Repository, dataset 186 "Wine Quality" (red wine part only).

- Page: https://archive.ics.uci.edu/dataset/186/wine+quality
- DOI: 10.24432/C56S3T
- Downloaded from: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- Download date: 2026-10-18
- License: CC BY 4.0

Paper to cite:

> Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
> Modeling wine preferences by data mining from physicochemical properties.
> Decision Support Systems, 47(4), 547-553.



## What the raw file is

- 1,599 rows of red wine, 12 columns (11 measurements + one quality score).
- Quality is an integer from 3 to 8, given by human tasters.
- No missing values.
- The file is semicolon separated (`;`), not comma separated. Reading it with
  the default comma gives one single column, so every read uses `sep=";"`.
- File as we downloaded it: `data/raw/winequality-red.csv`, 84,199 bytes,
  1,600 lines (1 header + 1,599 rows), md5 `2daeecee174368f8a33b82c8cccae3a5`.

## What we change

We never edit the raw file. Everything else is derived from it by
`ml/data_pipeline.py`.

The label: `quality >= 7` becomes 1 ("good"), everything below becomes 0.
Only 217 of the 1,599 wines are good, so the positive class is 13.6%.
That is why the models use `class_weight="balanced"` and why F1 is reported
next to accuracy.

v1 of the dataset (commit `bde3a70`):

- raw rows, duplicates kept
- 11 features
- split 1,279 train / 320 test, stratified

v2 of the dataset (commit `45c43f2`):

- 240 exact duplicate rows removed (1,599 -> 1,359)
- one new feature, `alcohol_density_ratio` = alcohol / density
- 12 features
- split 1,087 train / 272 test, stratified

Both splits use `random_state=42` and `test_size=0.2`. The scaler is fitted
inside the model pipeline, on the training split only, so the test rows never
influence preprocessing.

Duplicates matter here because the same row in both splits would make the test
score look better than it is. That is the main reason v2 exists.

## Which commit gives which data version

| Version | Commit    | DVC md5 (directory hash)                 | Files |
|---------|-----------|------------------------------------------|-------|
| raw     | `9c35744` | `f1c1b2e69091b72538a1cf7cde3fbb70.dir`   | 1     |
| v1      | `bde3a70` | `b11fcbef97cefe97662df7610dd3407d.dir`   | 2     |
| v2      | `45c43f2` | `5a8e5ef02766f26a2f2f9ada55364fa2.dir`   | 4     |

The md5 is the value inside the `.dvc` pointer file at that commit. The
version names v1 / v2 are only labels; the hash is the actual identity of the
data.

## What is in Git and what is not

In Git: the `.dvc` pointer files (`data/raw.dvc`, `data/processed.dvc`),
`data/.gitignore`, `data_pipeline.py`, and the training code.

Not in Git: the CSV files. They live in the DVC cache and on the Azure
remote (`myazure`, container `dvc`). `git ls-files` shows only pointers, so the
repository stays small and cloning it does not download 200 KB of data.

## How to get the data back

```powershell
dvc pull                                  # download current data from Azure
python ml/data_pipeline.py v1             # or rebuild from the raw file
git checkout bde3a70                      # go to the v1 commit
dvc checkout                              # restore the v1 files
```

`dvc checkout` after a `git checkout` is the important part: switching
commits changes the pointer, and `dvc checkout` makes the working files match
that pointer again.

## Known limits

- The dataset is small (1,599 rows) and from one region, so the model will not
  transfer to other wines.
- Quality is a subjective human score, not a measurement.
- The classes are imbalanced, and the positive class is small (217 rows).
  Metrics move a lot between splits, so we report precision, recall and F1
  and not accuracy alone.
