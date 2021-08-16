#!/bin/bash

## 训练 rl 模型
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
#    --write_model_dir ./result/dqn_p4_epoch800/run$i --planning_steps 4 \
#    --torch_seed $seed --validation_epoch_size 2000 --validation_span 40 --episodes 800 \
#    --device=cuda:3 --K_DQN 1 --to_do_what train_rl --log_filename train_rl.log
#done
#
# 生成对话数据
for ((i = 1; i <= 1; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch800/run$i \
    --torch_seed $seed --conversations_number_for_im_model 40 \
    --device=cuda:3 --to_do_what generate_state_action_pairs --log_filename generate_pairs.log
done

# 加载对话数据，训练 im 模型
for ((i = 1; i <= 1; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch800/run$i \
    --torch_seed $seed --validation_epoch_size 200 --conversations_number_for_im_model 40 \
    --device=cuda:3 --to_do_what train_im --log_filename train_im.log
done

# 验证模型效果
for ((i = 1; i <= 1; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch800/run$i \
    --torch_seed $seed --validation_epoch_size 200 \
    --device=cuda:3 --to_do_what validation --log_filename tmp.log
done
