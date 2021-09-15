#!/bin/bash




for ((i = 2; i <= 3; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
    --torch_seed $seed --early_stop_world_model True \
    --write_model_dir \
    ./result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet_stop_l_0.1_rate_0.9/run$i \
    --algorithm DPPO
done


for ((i = 2; i <= 3; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
    --torch_seed $seed \
    --write_model_dir ./result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet/run$i --algorithm DPPO
done