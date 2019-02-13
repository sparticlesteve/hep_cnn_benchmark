#!/bin/bash
#SBATCH -q debug
#SBATCH -C knl
#SBATCH -t 30
#SBATCH -J hep_train_tf

# Environment
module load tensorflow/intel-1.6.0-py36
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PWD:$PYTHONPATH

# Run
set -x
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u \
    python scripts/hep_classifier_tf_train_horovod.py \
    --config=configs/cori_knl_224_adam.json \
    --num_tasks=${SLURM_NNODES}
