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
Data is both the inference data after each run (no need to run the code yourself) and the DOE information.
Additionally, the test data is in the folder, named to the same scheme as the files above.

Traces (`*.nc`) are stored with Git LFS (`git lfs install` before cloning). `*.nc` is git-ignored, so new runs are committed on purpose with `git add -f data/<run>/`.

## License

MIT, see [LICENSE](LICENSE).
