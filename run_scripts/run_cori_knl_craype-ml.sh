#!/bin/bash
#SBATCH -q debug
#SBATCH -C knl
#SBATCH -t 30
#SBATCH -J hep_train_tf

# Environment setup
module load tensorflow/intel-1.6.0-py36
module use /global/cscratch1/sd/pjm/tmp_inst/modulefiles
module load craype-ml-plugin-py3/1.1.2
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PWD:$PYTHONPATH

# Run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u \
    python scripts/hep_classifier_tf_train_craype-ml.py \
    --config=configs/cori_knl_224.json \
    --num_tasks=${SLURM_NNODES}
