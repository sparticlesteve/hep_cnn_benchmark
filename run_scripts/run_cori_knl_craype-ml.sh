#!/bin/bash
#SBATCH -q debug
#SBATCH -C knl,quad,cache
#SBATCH -t 0:30:00
#SBATCH -J hep_train_tf

# Environment setup
module load tensorflow/intel-1.6.0-py27
module use /global/cscratch1/sd/pjm/tmp_inst/modulefiles
module load craype-ml-plugin-py2/1.1.2
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$PWD:$PYTHONPATH

# Binding settings
#bindstring="numactl -C 1-67,69-135,137-203,205-271"
bindstring=""

# Run the training
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u ${bindstring} \
    python scripts/hep_classifier_tf_train_craype-ml.py \
    --config=configs/cori_knl_224_adam.json \
    --num_tasks=${SLURM_NNODES}
#> hep_224x224_knl-craype-ml-plugin-2t_w$(( ${SLURM_NNODES} ))_p0.out 2>&1
