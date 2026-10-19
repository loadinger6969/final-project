# Data versions

The data is the UCI Wine Quality red set (1,599 rows, 11 measurements).
Full source, citation and licence are in `ml/PROVENANCE.md`. This file only
lists the two versions we train on and how they map to commits.

## v1

- raw rows as downloaded, duplicates kept (1,599 rows)
- 11 features
- split 1,279 train / 320 test, `random_state=42`, stratified
- commit `bde3a70`, DVC md5 `b11fcbef97cefe97662df7610dd3407d.dir`

## v2

- 240 exact duplicate rows removed (1,599 -> 1,359 rows)
- one added feature: `alcohol_density_ratio` = alcohol / density
- 12 features
- split 1,087 train / 272 test, same seed and stratify
- commit `45c43f2`, DVC md5 `5a8e5ef02766f26a2f2f9ada55364fa2.dir`

## What changed and why

| | v1 | v2 | reason |
|---|---|---|---|
| rows | 1,599 | 1,359 | 240 rows were exact copies |
| features | 11 | 12 | added alcohol / density |
| train / test | 1,279 / 320 | 1,087 / 272 | follows from the row count |


## Which commit has which version

| Version | Commit | DVC md5 | Files |
|---|---|---|---|
| raw | `9c35744` | `f1c1b2e69091b72538a1cf7cde3fbb70.dir` | 1 |
| v1 | `bde3a70` | `b11fcbef97cefe97662df7610dd3407d.dir` | 2 |
| v2 | `45c43f2` | `5a8e5ef02766f26a2f2f9ada55364fa2.dir` | 4 |

Both experiments in `README_experiments.md` use v2
(`5a8e5ef02766f26a2f2f9ada55364fa2.dir`), so the comparison is between models
and not between datasets.

## Switching versions

```powershell
git checkout bde3a70    # move the pointer back to v1
dvc checkout            # restore the v1 files into data/processed
```

To rebuild from the raw file instead:

```powershell
python ml/data_pipeline.py v1
python ml/data_pipeline.py v2
```
