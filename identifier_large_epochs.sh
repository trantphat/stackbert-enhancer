#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python ddp_identifier_large_epochs.py --nodes 1 --ngpus_per_node 3 > output.log 2>&1 &