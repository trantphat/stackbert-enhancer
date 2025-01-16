#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python ddp_identifier_train.py --nodes 1 --ngpus_per_node 4 --save_model