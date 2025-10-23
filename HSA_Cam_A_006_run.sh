#!/bin/bash
# Allocate one node
#SBATCH --nodes=1
# Number of program instances to be executed
#SBATCH --ntasks-per-node=1
# 8 GB memory required per node
#SBATCH --mem=8G
# Maximum run time of job
#SBATCH --time=2:00:00
# Give job a reasonable name
#SBATCH --job-name=HSA_Cam_A_006
# File name for standard output (%j will be replaced by job id)
#SBATCH --output=%x-%j.out
# File name for error output
#SBATCH --error=%x-%j.err
### this is a comment #SBATCH --gres=gpu:1

# Load software modules as needed, e.g.
module load devel/python/3.12.11
# module load devel/cuda/12.3
# module load lib/cudnn/9.0.0_cuda-12.3

unset PIP_TARGET

source ./pymcenv/bin/activate
# export PYTHONPATH="${PYTHONPATH}:./lib"

python3 HSA_Cam_A_006_jaxOp_run.py