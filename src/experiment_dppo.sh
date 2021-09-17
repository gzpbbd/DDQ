#!/bin/bash

n=5


# --- 20210916 验证 early stop 的效果
#
#n=5
## 不用 early stop, world model 5
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_warm_1000_run_300_simulate_1024_plan_5/run$i \
#     --algorithm DPPO  --episodes 300 --per_episode_time_steps 1024 --plan_k 5
#done
#
## 用 early stop  DPPO/DPPO_warm_1000_run_300_simulate_1024_plan_5_stop_l_10_rate_0.9
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir \
#     ./result/DPPO/DPPO_warm_1000_run_300_simulate_1024_plan_5_stop_l_10_rate_0.9/run$i \
#    --algorithm DPPO  --episodes 300 --per_episode_time_steps 1024 --plan_k 5 \
#    --early_stop_world_model True
#done
#
## 不用 world model, 500 轮，5次 PPO/PPO_warm_1000_run_500_simulate_1024
#
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/PPO/PPO_warm_1000_run_500_simulate_1024/run$i --algorithm PPO \
#    --episodes 500 --per_episode_time_steps 1024
#done

# 不用 world model, 500 轮，5次，数据量6倍，PPO/PPO_warm_1000_run_500_simulate_6144
# 运行耗时太长。运行一次需要 2 小时左右。还没运行
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/PPO/PPO_warm_1000_run_500_simulate_6144/run$i --algorithm PPO \
#    --episodes 500 --per_episode_time_steps 6144
#done


# --- 20210916 初步验证 world model net 中加入 attention 对 world model 对话成功率的影响
#n=1
# DPPO plan k =1, net=attention_r DPPO_run_100_wmnet_attention_r_e100_plan_1
# net attention_r
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_attention_r_sigmoid_e100_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type attention_r --warm_episodes 1000
#done


# DPPO plan k =1, net=original DPPO_run_100_wmnet_original_e30_plan_1
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_100_wmnet_original_e100_plan_1/run$i \
#     --algorithm DPPO  --episodes 100 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original --warm_episodes 1000
#done


# 20210917 验证 world model 网络结构 h+a -> cat(h,a)

# net attention_r_cat
#n=1
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_attention_r_cat_sigmoid_e100_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type attention_r_cat --warm_episodes 1000
#done


#
## net attention_r train epoch 30
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_attention_r_e30_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type attention_r --warm_episodes 1000  --world_model_train_epochs 30
#done
#
## net original train epoch 30
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_original_e30_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original --warm_episodes 1000 --world_model_train_epochs 30
#done

# net original relu train 100
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_100_wmnet_original_relu_e100_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original_relu --warm_episodes 1000 --world_model_train_epochs 100
#done

# net original more fc
#
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_100_wmnet_original_more_fc_e100_plan_1/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original_more_fc --warm_episodes 1000 --world_model_train_epochs 100
#done

## original train e 100
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_100_wmnet_original_e100_plan_1_again/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original --warm_episodes 1000 --world_model_train_epochs 100
#done

# add a common attention layer
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_e100_plan_1_add_attention_layer/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type original --warm_episodes 1000 --world_model_train_epochs 100
#done

# attention reward + dropout
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_e100_plan_1_add_attention_r_dropout/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type attention_r_dropout --warm_episodes 1000 --world_model_train_epochs 100
#done


# attention layer + dropout
#for ((i = 1; i <= n; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
#    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
#    --torch_seed $seed \
#    --write_model_dir ./result/DPPO/DPPO_run_40_wmnet_e100_plan_1_attention_layer_dropout/run$i \
#     --algorithm DPPO  --episodes 40 --per_episode_time_steps 1024 --plan_k 1 \
#     --world_model_net_type attention_layer_dropout --warm_episodes 1000 --world_model_train_epochs 100
#done

# 20210917
# 重做 early stop 实验
for ((i = 1; i <= n; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
    --torch_seed $seed \
    --write_model_dir \
     ./result/DPPO/DPPO_warm_1000_run_300_simulate_1024_plan_5_stop_l_10_rate_0.9_correct/run$i \
    --algorithm DPPO  --episodes 300 --per_episode_time_steps 1024 --plan_k 5 \
    --early_stop_world_model True
done


# 做 PPO-K5 实验
# 不用 world model, 500 轮，5次，数据量6倍，PPO/PPO_warm_1000_run_500_simulate_6144
# 运行耗时太长。运行一次需要 2 小时左右。还没运行
for ((i = 1; i <= n; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python -u  \
    /home/huangchenping/github_repo/DDQ-master/src/run_dppo.py \
    --torch_seed $seed \
    --write_model_dir ./result/PPO/PPO_warm_1000_run_500_simulate_6144/run$i --algorithm PPO \
    --episodes 500 --per_episode_time_steps 6144
done
