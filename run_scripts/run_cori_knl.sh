#!/bin/bash
#SBATCH -q debug
#SBATCH -C knl
#SBATCH -t 30
#SBATCH -J hep_train_tf

# Set up environment
module load tensorflow/intel-1.6.0-py36
export PYTHONPATH=$PWD:$PYTHONPATH

# Configure number of parameter servers
if [ $SLURM_NNODES -ge 2 ]; then
    NUM_PS=1

    # Check if user specified more PS
    while getopts p: option
    do
	case "${option}"
	in
	p) NUM_PS=${OPTARG};;
	esac
    done

    if [ ${NUM_PS} -ge ${SLURM_NNODES} ]; then
	echo "The number of nodes has to be bigger than the number of parameters servers"
	exit
    fi

else
    NUM_PS=0
fi

# Run the training
set -x
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 272 -u \
    python scripts/hep_classifier_tf_train.py \
    --config=configs/cori_knl_224.json \
    --num_tasks=${SLURM_NNODES} \
    --num_ps=${NUM_PS}
