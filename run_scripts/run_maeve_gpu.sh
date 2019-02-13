#!/bin/bash

#run
export PYTHONPATH=$PWD:$PYTHONPATH

#set some parameters
export CUDA_VISIBLE_DEVICES=0

#run the stuff
python scripts/hep_classifier_tf_train.py --config=configs/maeve_gpu_224.json --num_tasks=1 --num_ps=0
