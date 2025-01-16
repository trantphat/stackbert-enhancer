#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python ddp_classifier_tune.py --nodes 1 --ngpus_per_node 4 > output.log 2>&1 &

# nohup python ddp_k_target.py --nodes 1 --ngpus_per_node 4 > output.log 2>&1 &
