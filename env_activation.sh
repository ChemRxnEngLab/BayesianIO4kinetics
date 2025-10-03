# Load software modules as needed, e.g.
module load devel/python/3.13.3
module load devel/cuda/12.3
module load lib/cudnn/9.0.0_cuda-12.3

unset PIP_TARGET

source ./pymcenv/bin/activate
export PYTHONPATH="${PYTHONPATH}:./lib"