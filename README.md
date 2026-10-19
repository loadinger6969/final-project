# Wine Quality Predictor

A small ML app that guesses if a red wine is "good" from its lab
measurements. We trained the model ourselves on the UCI Wine Quality
dataset, and wrapped the whole thing in DevOps and MLOps tooling.

## What it does

Send 11 measurements of a wine to `/predict` and the API answers with
`good` or `not good`, plus a probability. We call a wine good when its
quality score is 7 or 8.

## Run it locally

```bash
pip install -r requirements-dev.txt
python ml/data_pipeline.py
python ml/train.py v2 baseline
python ml/train.py v2 improved
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs.

## Run the tests

```bash
pytest -v
```

## Run it in Docker

```bash
docker build -t wine-app .
docker run -p 8000:8000 wine-app
```

## Branches and flow

We use four levels:

- `feature/*` – day-to-day work, one branch per change
- `dev` – features meet here
- `staging` – pre-release check
- `main` – the release

The path is always `feature -> dev -> staging -> main`. Every pull
request to dev/staging/main runs CI: lint, format, tests, and a Docker
build. Merging to staging builds a staging image with staging secrets.
Merging to main builds a production image with production secrets.

1. Commit on a feature branch
2. Open a PR to `dev` -> CI builds and tests
3. PR to `staging` -> staging image
4. PR to `main` -> production image

## Structure

```text
app/                    FastAPI service
ml/
  data_pipeline.py      raw csv -> v1/v2 datasets
  train.py              trains + logs to MLflow
  configs/
  registry/             the model we serve
  README_data.md        data versions
  README_experiments.md experiment results
  PROVENANCE.md         where the data comes from
data/                   DVC-tracked (only .dvc pointers in git)
.github/workflows/      ci.yml, deploy.yml
Dockerfile, .dockerignore
DEPLOYMENT.md, README.md
```

## Team

- `loadinger6969` – app, tests, Docker, CI/CD
- `sblzc` – data, DVC, training, MLflow
