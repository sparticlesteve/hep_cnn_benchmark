#!/bin/bash

#set some parameters
export CUDA_VISIBLE_DEVICES=0

#run the stuff
python train.py --config=configs/maeve_gpu_224.json --num_tasks=1 --num_ps=0
