#!/bin/bash
# Local run of the inference-free BOED loop (HSA_K_001) on a small machine.
#
# What is left to parallelise
# ---------------------------
# The per-candidate MCMC refits and the nested process pools are GONE. What
# remains is one real-data MCMC per design iteration (32 per seed, 8 chains).
# That is now the only thing that benefits from parallelism, and it benefits a
# lot -- numpyro only draws chains in parallel when the XLA host device count is
# >= num_chains (8). Measured here on 6 cores, n_obs=32:
#
#     DEVICES=8  ->   6.7 s per MCMC        <-- default
#     DEVICES=6  ->  34.9 s  (silently falls back to sequential: 6 < 8 chains)
#     DEVICES=1  ->  35.0 s  (strictly sequential)
#
# Oversubscribing 8 devices onto 6 cores is 5x faster than "sequential" and
# costs nothing but a little memory, so DEVICES defaults to 8. Set DEVICES=1
# if you really want one chain at a time.
#
# Usage:
#   ./HSA_K_001_local.sh                          # 8 chain-devices, 1 BLAS thread
#   DEVICES=1 NTHREADS=1 ./HSA_K_001_local.sh     # strictly sequential (~5x slower)
#   NTHREADS=6 ./HSA_K_001_local.sh               # let BLAS/torch use all 6 cores
#   PYTHON=~/Environments/foo/bin/python3 ./HSA_K_001_local.sh
#   ITERATIONS=2 OPTITER=3 ./HSA_K_001_local.sh   # quick shakedown
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
DEVICES="${DEVICES:-8}"     # XLA host devices; match num_chains=8 for parallel chains
NTHREADS="${NTHREADS:-1}"   # BLAS / torch / XLA intra-op threads
ITERATIONS="${ITERATIONS:-32}"
OPTITER="${OPTITER:-30}"
TRIALS="${TRIALS:-4}"

export BIBO_HOST_DEVICES="${DEVICES}"   # read by HSA_K_001.py before jax is imported
export JAX_PLATFORMS=cpu                # stay off the GPU
export OMP_NUM_THREADS="${NTHREADS}"    # torch / BoTorch, i.e. the Ax surrogate
export MKL_NUM_THREADS="${NTHREADS}"
export OPENBLAS_NUM_THREADS="${NTHREADS}"
export NUMEXPR_NUM_THREADS="${NTHREADS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=${NTHREADS}"

# --- preflight: ax-platform is easy to miss locally -------------------------
if ! "$PYTHON" - <<'PY'
import importlib.util as u, sys
missing = [m for m in ("ax", "numpyro", "jax", "arviz", "scipy", "xarray", "namer")
           if u.find_spec(m) is None]
if missing:
    sys.stderr.write("Missing modules: %s\n" % ", ".join(missing))
    sys.exit(1)
PY
then
    echo "ERROR: interpreter '$PYTHON' cannot run HSA_K_001.py (see above)." >&2
    echo "Point PYTHON= at a suitable environment, e.g." >&2
    echo "  PYTHON=~/Environments/TVBOvenv/bin/python3 $0" >&2
    echo "NOTE: ax-platform is missing from every local interpreter as of writing." >&2
    echo "      Install it with:  pip install 'ax-platform==1.2.3'" >&2
    exit 1
fi

echo "HSA_K_001 local run: DEVICES=${DEVICES} NTHREADS=${NTHREADS} python=${PYTHON}"

# Ax refits its GP surrogate once per --optiterations and is the dominant cost
# now that inference is cheap. Same 120 candidates with 2.5x fewer GP fits:
#   OPTITER=12 TRIALS=10 ./HSA_K_001_local.sh
exec "$PYTHON" HSA_K_001.py \
    --n-seeds 1 \
    --seed 43 \
    --iterations "${ITERATIONS}" \
    --optiterations "${OPTITER}" \
    --trialsperiteration "${TRIALS}" \
    --n-outer 512 \
    --n-inner 4096 \
    --prior-samples 16384 \
    --datadir "$(pwd)/data"
