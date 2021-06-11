#!/bin/bash
#Below is the script used for figure 4

# 跨服务器拷贝实验结果
# scp -r huangchenping@10.12.41.220:/home/huangchenping/github_repo/DDQ-master/src/result/improve* .

# 对每种模型都运行了五次，每次设置 pytorch 的随机数种子不一样，实验结果分别保存
# 每次运行改变的参数：write_model_dir，planning_steps，torch_seed

# improve replay pool: DDQ 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 800 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/improve/DDQ_k5_run$i \
#    --planning_steps 4 --torch_seed $seed --log_string ./result/improve/DDQ_run$i/train_k5.log \
#    --improve_replay_pool true
#done

# improve replay pool and just train on newly experiment: DDQ 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/improve_td_err-newly_experience/run$i \
#    --planning_steps 4 --torch_seed $seed --improve_replay_pool true
#done

## just train on newly experiment: DDQ 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/improve-newly_experience/run$i \
#    --planning_steps 4 --torch_seed $seed
#done

## just train with double sample: DDQ 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/improve-newly_experience/run$i \
#    --planning_steps 4 --torch_seed $seed --improve_replay_pool true
#done

# just train with double sample: DDQ 5
for ((i = 1; i <= 3; i++)); do
  let "seed=$i*100"
  python run.py --agt 9 \
    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
    --planning_steps 4 --torch_seed $seed  \
    --write_model_dir ./result/reduce_world_pool-td_err_sample/run$i \
    --agent_pool_size_for_wor_exp 1500  --improve_replay_pool true
done

## train more epochs on experience
#for ((i = 1; i <= 3; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --planning_steps 4 --torch_seed $seed  \
#    --write_model_dir ./result/train_on_experience_for_more_epoches/run$i \
#    --agent_pool_size_for_wor_exp 1500 --agent_train_epochs 3
#done

## change replay pool size: DQN 1
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 800 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/change_pool_size_20aveturn_800_epoches/run$i \
#    --planning_steps 0 --torch_seed $seed \
#    --log_string ./result/change_pool_size_20aveturn_800_epoches/run$i/train_k5.log \
#    --improve_replay_pool true
#done

#
### baseline DDQ 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 800 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/DDQ_change_pool_size_20_800_episodes/run$i \
#    --planning_steps 4 --torch_seed $seed --log_string ./result/DDQ_change_pool_size_20_800_episodes/run$i/train.log
#done

## baseline DQN 5
#for ((i = 1; i <= 5; i++)); do
#  let "seed=$i*100"
#  python run.py --agt 9 \
#    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
#    --experience_replay_pool_size 5000 --episodes 800 --simulation_epoch_size 100 \
#    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
#    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
#    --warm_start 1 --warm_start_epochs 100 --grounded 1 --boosted 1 --train_world_model 1 \
#    --write_model_dir ./result/baseline_dqn_k5_5times/run$i \
#    --planning_steps 4 --torch_seed $seed --log_string ./result/baseline_dqn_k5_5times/run$i/train.log
#done
