#!/bin/bash

# ---------------- SLURM Job Settings ----------------

#SBATCH --job-name=RNA_train                # Job name for identification
#SBATCH --partition=all_gpu                 # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --gres=gpu:a100:4                   # Request a100
##SBATCH --ntasks-per-node=1                # Number of tasks (processes) per node

#SBATCH -t 4-0:00                           # Max wall time: 4 days
#SBATCH -o run_outputs.out                        # File to write standard output (%j = job ID)
#SBATCH -e run_error.err                        # File to write standard error (%j = job ID)

# ---------------- Environment Setup ----------------

# Load the appropriate module (with CUDA-aware MPI already built)
module load conda/24.11.1
conda activate env_cuda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Makes the paths from conda environment visible
# ---------------- Job Execution --------------------

# Run the simulation using MPI with 4 processes
python -u main_Train.py
