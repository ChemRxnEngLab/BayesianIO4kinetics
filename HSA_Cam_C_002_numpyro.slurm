#!/bin/bash
# Allocate one node
#SBATCH --nodes=1
# Number of program instances to be executed
#SBATCH --ntasks-per-node=1
# Number of CPUs per task
#SBATCH --cpus-per-task=48
# 128 GB memory required per node
#SBATCH --mem=128G
# Maximum run time of job
#SBATCH --time=14:00:00
# Give job a reasonable name
#SBATCH --job-name=HSA_Cam_C_002_numpyro
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

source ./numpyroenv/bin/activate
# export PYTHONPATH="${PYTHONPATH}:./lib"

python3 HSA_Cam_C_002_numpyro.py