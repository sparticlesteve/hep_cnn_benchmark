#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl,quad,cache
#SBATCH -t 0:30:00
#SBATCH -J hep_train_tf

# DOESN'T WORK CURRENTLY. NEEDS UPDATE

#use custom craype-ml installation
module use /global/homes/t/tkurth/custom_rpm

#set up python stuff
module load tensorflow/intel-head
module use /global/homes/t/tkurth/custom_rpm/modulefiles
module load craype-ml-plugin-py2/1.1.0

#better binding
#bindstring="numactl -C 1-67,69-135,137-203,205-271"
bindstring=""

#run
cd ../scripts/

#launch srun
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u ${bindstring} \
    python hep_classifier_tf_train_craype-ml.py \
    --config=configs/cori_knl_224_tune.json \
    --num_tasks=${SLURM_NNODES}

#> hep_224x224_knl-craype-ml-plugin-2t_w$(( ${SLURM_NNODES} ))_p0.out 2>&1
