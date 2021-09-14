#!/bin/bash


# just train with double sample: DDQ 5
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
    --torch_seed $seed \
    --write_model_dir ./result/DPPO_warm_1000_run_300/run$i --algorithm DPPO
done