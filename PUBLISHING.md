# Publishing vllm-benchmark

This directory is an independent Git repository. The distribution is named
`vllm-benchmark`, the Python package and command are `vllm_bench`, and the source
repository is `viniciusCSreis/vllm_benchmark`. The project uses the MIT license.
No sibling checkout is needed to build, test, install, or publish it.

## Create the public repository

Create an empty public GitHub repository named `vllm_benchmark` under
`viniciusCSreis`. Do not initialize it with a README or license; these files
already exist locally. From this directory:

```bash
git add .
git commit -m "Prepare standalone benchmark package"
git remote add origin https://github.com/viniciusCSreis/vllm_benchmark.git
git push -u origin HEAD
```

If `origin` already exists, inspect `git remote -v` and use `git remote set-url`
instead of adding it again. Update `[project.urls]` in `pyproject.toml` if you
choose another owner or repository name.

## Check a release locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
python -m unittest discover -s tests
python tests/compose_smoke.py --image python:3.12-slim
python -m build
python -m twine check dist/*
```

The Compose smoke check requires Docker and launches ordinary Python containers;
it does not load a model or allocate cloud GPUs. The build creates a wheel and
source distribution in `dist/`. Use an empty `dist/` for each release so older
versions are not uploaded with it.

The version is defined only in `vllm_bench/__init__.py`; the initial release is
`0.1.0`. Change it before every later release. PyPI does not allow reusing an
uploaded version's filenames.

## Configure PyPI Trusted Publishing

1. Create a PyPI account, verify your email, and enable two-factor authentication.
2. In GitHub repository Settings, create an Actions environment named `pypi`.
3. Open <https://pypi.org/manage/account/publishing/> and add a pending GitHub
   publisher with these exact values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `vllm-benchmark` |
   | Owner | `viniciusCSreis` |
   | Repository | `vllm_benchmark` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

The package name had no public PyPI project during preparation; PyPI still must
accept the name when you register/publish. If it is unavailable, choose another
`[project].name` and update this guide, install commands, and the workflow URL.
The import name can remain `vllm_bench`.

See PyPI's [pending publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [GitHub Actions publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
The workflow authenticates with GitHub OIDC, so no PyPI API-token secret is needed.

## Publish the release

Wait for CI to pass, then create and push a version tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

On GitHub, create and publish a Release using that existing tag. Publishing the
release triggers `.github/workflows/publish.yml`; pushing a tag alone does not.
The workflow runs checks, verifies the tag matches the package version, builds
both distributions, and uploads them to PyPI through the `pypi` environment.
If you configured environment reviewers, approve the waiting publish job.

After the workflow succeeds, check installation in a fresh virtual environment:

```bash
python3 -m venv /tmp/vllm-benchmark-check
/tmp/vllm-benchmark-check/bin/python -m pip install vllm-benchmark
/tmp/vllm-benchmark-check/bin/vllm_bench --version
```

Users install Modal support with `python -m pip install 'vllm-benchmark[modal]'`.
For later releases, update the version, commit and push, then repeat with the
matching `vX.Y.Z` tag and GitHub Release.
