#!/bin/bash

# 生成对话数据
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --torch_seed $seed --conversations_number_for_im_model 50 \
    --device=cuda:3 --to_do_what generate_state_action_pairs --log_filename generate_pairs.log
  sleep 5
done

# 加载对话数据，训练 im 模型
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --torch_seed $seed --validation_epoch_size 10 --conversations_number_for_im_model 50 \
    --device=cuda:3 --to_do_what train_im --filename_of_im_model im_model_50.pkl \
    --log_filename train.log
  sleep 5
done

# 验证模型效果
for ((i = 1; i <= 5; i++)); do
  let "seed=$i*100"
  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
    --write_model_dir ./result/dqn_p4_epoch400/run$i \
    --torch_seed $seed --validation_epoch_size 300 \
    --filename_of_im_model im_model_50.pkl \
    --filename_of_rl_model rl_model.pkl \
    --device=cuda:3 --to_do_what validation --log_filename validation.log
  sleep 5
done

## 训练 rl 模型 with shaping
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
#    --write_model_dir ./result/dqn_p4_epoch400/run$i --planning_steps 4 \
#    --torch_seed $seed --validation_epoch_size 1000 --validation_span 20 --episodes 400 \
#    --device=cuda:3 --K_DQN 1 --to_do_what train_rl_with_shaping \
#    --filename_of_rl_model rl_model_with_shaping.pkl \
#    --filename_of_im_model im_model_40.pkl \
#    --filename_of_performances performance_with_shaping.json \
#    --log_filename train_rl_with_shaping.log
#sleep 5
#done
#
## 训练 rl 模型
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  /home/huangchenping/software/miniconda3/envs/DDQ_/bin/python run.py \
#    --write_model_dir ./result/dqn_p4_epoch800/run$i --planning_steps 4 \
#    --torch_seed $seed --validation_epoch_size 2000 --validation_span 40 --episodes 800 \
#    --device=cuda:3 --K_DQN 1 --to_do_what train_rl --log_filename train_rl.log
#sleep 5
#done
