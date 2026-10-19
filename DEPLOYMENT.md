# Deployment

We do not push images to a real registry in this project. Instead, each
environment is built by a GitHub Actions workflow, and each uses its own
secrets.

## Environments

| Branch    | Image tag       | Secret used          |
|-----------|-----------------|----------------------|
| `staging` | `wine-app:staging` | `STAGING_API_KEY` |
| `main`    | `wine-app:production` | `PRODUCTION_API_KEY` |

## Set the secrets

On GitHub: **Settings -> Secrets and variables -> Actions -> New
repository secret**.

- Add `STAGING_API_KEY` with a value for staging.
- Add `PRODUCTION_API_KEY` with a value for production.

## What happens on a merge

When a pull request is merged to `staging`, `deploy.yml` runs. It writes
a `.env` file using the staging secret and builds `wine-app:staging`.
When one is merged to `main`, it does the same with the production secret
and builds `wine-app:production`.

## Protect the branches

**Settings -> Branches -> Add branch protection rule** for `dev`,
`staging`, and `main`. Require a pull request and require the CI check to
pass. This is what stops a red branch from being merged.

## Build an image by hand

```bash
# staging
docker build -t wine-app:staging .
# production
docker build -t wine-app:production .

# run either one
docker run -p 8000:8000 wine-app:staging
```
