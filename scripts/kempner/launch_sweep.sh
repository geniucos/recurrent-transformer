#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --output=logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=150GB		
#SBATCH --partition=<PARTITION>
#SBATCH --array=0-3

# Activate environment
source ~/.bashrc
conda deactivate
conda activate recurrent
cd <REPO_ROOT>

export CONFIG=configs/kempner/base-c4-t5.yaml+configs/kempner/models/150m.yaml
export SWEEP_CONFIG=configs/kempner/sweeps/cosine_default.yaml
export CHECKPOINTS_PATH=<CHECKPOINTS_DIR>

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/kempner/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}
