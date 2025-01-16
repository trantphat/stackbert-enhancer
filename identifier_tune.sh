#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python ddp_identifier_tune.py --nodes 1 --ngpus_per_node 4

# nohup python ddp_k_target.py --nodes 1 --ngpus_per_node 4 > output.log 2>&1 &
