#!/bin/bash

# 训练 rl 模型
for ((i = 5; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --torch_seed $seed --validation_epoch_size 300 --validation_span 10 --episodes 400 \
    --planning_steps 4 --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --device=cuda:2 --K_DQN 1  --to_do_what train_rl
done

# 生成对话数据
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --torch_seed $seed  --conversations_number_for_im_model 100000 \
    --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --device=cuda:2 --to_do_what generate_state_action_pairs
done

# 加载对话数据，训练 im 模型
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --torch_seed $seed --validation_epoch_size 300 \
    --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --device=cuda:2 --to_do_what train_im
done


