#!/bin/bash

for ((i = 1; i <= 3; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --torch_seed $seed --validation_epoch_size 300 --validation_span 30 --episodes 1000 \
    --planning_steps 0 --write_model_dir ./result/dqn_p0_epoch1500/run$i --policy_method rl \
    --device=cuda:3
done
