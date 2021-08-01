#!/bin/bash
#Below is the script used for figure 4

# 跨服务器拷贝实验结果
# scp -r huangchenping@10.12.41.220:/home/huangchenping/github_repo/DDQ-master/src/result/double_dqn .

# 对每种模型都运行了五次，每次设置 pytorch 的随机数种子不一样，实验结果分别保存
# 每次运行改变的参数：write_model_dir，planning_steps，torch_seed

# train using Double DQN
for ((i = 1; i <= 3; i++)); do
  let "seed=$i*100"
  python run.py --agt 9 \
    --usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
    --experience_replay_pool_size 5000 --episodes 400 --simulation_epoch_size 100 \
    --run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
    --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
    --warm_start 1 --warm_start_epochs 100 --grounded 0 --boosted 1 --train_world_model 1 \
    --planning_steps 4 --torch_seed $seed  \
    --write_model_dir ./result/double_dqn-reduce_pool/run$i \
    --agent_pool_size_for_wor_exp 1500 --dqn_variant double_dqn
done

