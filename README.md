# Overview for repo

This repository assumes a Linux setup with Python 3.12.

## Environment

The environment is managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists the direct dependencies, `uv.lock` pins the full environment.

```bash
uv sync                    # runtime + notebook dependencies
uv sync --extra cuda       # additionally the CUDA 12 build of jax
uv run python HSA_K_003.py --help
```

`requirements.txt` is generated from `uv.lock` for plain-pip setups (e.g. SLURM clusters):

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt
```

Regenerate it whenever `uv.lock` changes.

The `*.slurm` jobs activate `./.venv`. Create it once on the cluster, with uv or with pip:

```bash
module load devel/python/3.12.11
uv sync --no-default-groups                  # or:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Naming scheme

|Pattern | Content |
|--|--|
|`HSA_<campaign>_<nnn>.py` / `.ipynb` / `.slurm` | experiment script, its evaluation notebook and cluster job (e.g. `HSA_B_042`, `HSA_K_003`) |
|`HSA_<campaign>_Overview*.ipynb` | comparison of all runs of a campaign |
|`HSA_Mo_<x>_<nnn>.py` | Bayesian models |
|`HSA_GTMo_<x>_<nnn>.py` | ground-truth models for synthetic data |
|`data/HSA_Da_<x>_<nnn>.ipynb` | generation of the test data `data/test_HSA_Da_<x>_<nnn>.nc` |

Older files used a `_numpyro` suffix and a `Cam_` infix (e.g. `HSA_Cam_B_042_numpyro.py` is now `HSA_B_042.py`). The run metadata in `data/<run>/experiment_summary_*.json` still records the original file names.

## available scripts

The jupyter notebooks are for visualization while the .py files are for running the actual code. Can be run on SLURM Clusters for better parallelization.

|Script | Description | Author | Data source |
|--|--|--|--|
|B_036 | Kinetic parameter fitting | Koschany DOE | data/chivalrous-infographic |
|B_039 | Kinetic parameter fitting | paramEKLD bDOE | data/agitated-health |
|B_042 | Kinetic parameter fitting | predEKLD bDOE | data/eternal-rock |
|B_042_Overview | Kinetic parameter fitting | all DOEs | all data s.a. |
|D_007 | Deactivation Modeling | predEMD bDOE | data/icky-link |

## lib files

|File | Description |
|--|--|
Mo_B_011 | Kinetic parameter fitting |
Mo_B_012 | Kinetic parameter fitting, informative priors |
Mo_D_001 | Deactivation Modeling Mo1|
Mo_D_002 | Deactivation Modeling Mo2|
Mo_D_003 | Deactivation Modeling Mo3|
Mo_D_004 | Deactivation Modeling Mo4|
GTMo_C_001 | Deactivation Modeling Ground truth Mo1 |
GTMo_C_002 | Deactivation Modeling Ground truth Mo2 |
GTMo_C_003 | Deactivation Modeling Ground truth Mo3 |
GTMo_C_004 | Deactivation Modeling Ground truth Mo4 |
NestPools | Utility for running recursively parallelized (nested) pools on linux machines|

## data
Each run writes to `data/<run>/`: the DOE information and run metadata (`*.json`) are tracked in git.
The synthetic test data `data/test_HSA_Da_<x>_<nnn>.nc` is tracked as well.

The inference traces (`data/<run>/**/*.nc`, about 22 GB) are **not** part of the repository: they are git-ignored, kept locally and will be published separately (e.g. on Zenodo).
`data/traces_manifest.csv` lists every trace with size and SHA-256; check a local or downloaded copy with

```bash
tail -n +2 data/traces_manifest.csv | awk -F, '{print $3"  "$1}' | sha256sum -c --quiet
```

## License

MIT, see [LICENSE](LICENSE).
