#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl,quad,cache
#SBATCH -t 0:30:00
#SBATCH -J hep_train_tf

#*** License Agreement ***
#
# High Energy Physics Deep Learning Convolutional Neural Network Benchmark
# (HEPCNNB) Copyright (c) 2017, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# (1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# (2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# (3) Neither the name of the University of California, Lawrence Berkeley
#     National Laboratory, U.S. Dept. of Energy nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to Lawrence Berkeley National
# Laboratory, without imposing a separate written license agreement for such
# Enhancements, then you hereby grant the following license: a non-exclusive,
# royalty-free perpetual license to install, use, modify, prepare derivative
# works, incorporate into other computer software, distribute, and sublicense
# such enhancements or derivative works thereof, in binary and source code form.
#---------------------------------------------------------------      


# TODO : THIS SCRIPT NEEDS TO BE UPDATED
source /cray/css/users/jbalma/bin/setup_env_cuda9.sh
source /cray/css/users/jbalma/bin/env_python.sh

#use custom craype-ml installation
#module use /global/homes/t/tkurth/custom_rpm

#set up python stuff
#module load tensorflow/intel-head
#module use /global/homes/t/tkurth/custom_rpm/modulefiles
module load craype-ml-plugin-py2 

#better binding
#bindstring="numactl -C 1-67,69-135,137-203,205-271"
bindstring=""

#run
PPN=1
NODES=1
NP=1
export OMP_NUM_THREADS=18
export SLURM_WORKING_DIR=/lus/scratch/jbalma/temp/junk_hepcnn_run
mkdir ${SLURM_WORKING_DIR}
cd ../
echo ${PWD}
cp -r ./* ${SLURM_WORKING_DIR}/
cd ${SLURM_WORKING_DIR}/
export PYTHONPATH="$PYTHONPATH:${PWD}:${PWD}/scripts:${PWD}/slurm_tf_helper"
echo ${PWD}

srun -N ${NODES} -n ${NP} -c ${OMP_NUM_THREADS} -C P100 --gres=gpu -u python scripts/hep_classifier_tf_train_craype-ml.py --config=configs/xc50_gpu_224.json --num_tasks=${NP} --dummy_data |& tee ./hep_224x224_p100-craype-ml-plugin-2t_${NODES}_p0.out #2>&1



