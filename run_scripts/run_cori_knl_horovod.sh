#!/bin/bash
#SBATCH -q premium
#SBATCH -C knl
#SBATCH -t 1:00:00
#SBATCH -J hep_train_tf

# Environment
module load tensorflow/intel-1.8.0-py27
export PYTHONPATH=$PWD:$PYTHONPATH

# Run
set -x
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u \
    python scripts/hep_classifier_tf_train_horovod.py \
    --config=configs/cori_knl_224_adam.json \
    --num_tasks=${SLURM_NNODES}
