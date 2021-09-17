# encoding:utf-8

import os
import argparse
import json

import logging
from matplotlib import pyplot
import pandas as pd
import seaborn as sns
from deep_dialog.utils import SmoothValue, calculate_time
from collections import OrderedDict

logging.getLogger().setLevel(logging.DEBUG)
pyplot.figure(dpi=1000)


def calculate_average_success_rate(result_dir, filename):
    # 获取各个文件
    all_result = []
    sub_dir_list = os.listdir(result_dir)
    for sub_dir in sub_dir_list:
        sub_dir_path = os.path.join(result_dir, sub_dir)
        file_path = os.path.join(sub_dir_path, filename)

        with open(file_path, 'r') as f:
            result = json.load(f)
            all_result.append(result['success_rate'])

    # 综合结果计算均值
    average_success_rate = {}
    for success_rate in all_result:
        for epoch, rate in success_rate.items():
            average_success_rate[int(epoch)] = average_success_rate.get(int(epoch), 0) + rate

    epochs = average_success_rate.keys()
    average_rate = [average_success_rate[epoch] / len(all_result) for epoch in epochs]
    return average_rate


def draw_figures(dirs):
    ddq = calculate_average_success_rate(result_dir='baseline/baseline_ddq_k5_5_agent_800_epoches/',
                                         filename='train_performance.json')
    dqn = calculate_average_success_rate(result_dir='baseline/baseline_dqn_k5_5_agent_800_epoches',
                                         filename='train_performance.json')

    # 画图
    epochs = list(range(1, len(ddq) + 1))
    pyplot.plot(epochs, ddq, label='DDQ 5')
    pyplot.plot(epochs, dqn, label='DQN 5')

    for dir in dirs:
        result = calculate_average_success_rate(
            result_dir=os.path.join('result', dir),
            filename='train_performance.json')
        pyplot.plot(epochs, result, label=dir)

    pyplot.xlabel('episode')
    pyplot.ylabel('success rate')
    pyplot.legend()
    pyplot.show()


@calculate_time
def read_to_data_frame(result_dir, algorithm, filename):
    df = pd.DataFrame(columns=['algorithm', 'run_num', 'epoch', 'success_rate', 'environment'])
    sub_dir_list = os.listdir(result_dir)
    for sub_dir in sub_dir_list:  # 对每个子目录的文件
        file_path = os.path.join(result_dir, sub_dir, filename)
        with open(file_path, 'r') as f:
            result = json.load(f)  # 读取记录
            # 加入 data frame 中

            # 读取 user suc
            smooth = SmoothValue(5)
            result['success_rate'] = OrderedDict(
                sorted(result['success_rate'].items(), key=lambda (key, value): int(key)))
            for epoch, success_rate in result['success_rate'].items():
                success_rate = smooth.smooth(success_rate)
                df = df.append([{'algorithm': algorithm, 'run_num': sub_dir, 'epoch': int(epoch),
                                 'success_rate': success_rate, 'environment': 'user'}])

            # 读取 world model suc
            if 'success_rate_with_world_model' not in result.keys():
                continue
            smooth = SmoothValue(5)
            result['success_rate_with_world_model'] = OrderedDict(
                sorted(result['success_rate_with_world_model'].items(),
                       key=lambda (key, value): int(key)))
            for epoch, success_rate in result['success_rate_with_world_model'].items():
                success_rate = smooth.smooth(success_rate)
                df = df.append([{'algorithm': algorithm, 'run_num': sub_dir, 'epoch': int(epoch),
                                 'success_rate': success_rate, 'environment': 'world_model'}])
    return df


@calculate_time
def draw_figure_from_data_frame(result_dirs, filename='performance.json'):
    all_df = []
    for dir in result_dirs:
        print 'Read data from {}'.format(os.path.abspath(dir['dir']))
        result_df = read_to_data_frame(dir['dir'], dir['algorithm'], filename)
        all_df.append(result_df)
        print len(result_df)
    df = pd.concat(all_df)
    df.sort_values(by='run_num', inplace=True)

    selected_df = df
    # 筛选
    # selected_df = selected_df[(selected_df['epoch'] < 150)]
    # selected_df = selected_df[(selected_df['environment'] == 'user')]
    # 合并 environment 与 algorithm
    # dppo_df = df[df['algorithm'].str.startswith('DPPO')]
    # ppo_df = df[df['algorithm'].str.startswith('PPO')]

    # dppo_df['algorithm_environment'] = dppo_df['algorithm'] + '_' + dppo_df['environment']
    # ppo_df['algorithm_environment'] = ppo_df['algorithm']
    # selected_df = pd.concat([dppo_df, ppo_df])
    # sns.relplot(x='epoch', y='success_rate', hue='algorithm_environment',
    #             data=selected_df, kind='line')

    sns.relplot(x='epoch', y='success_rate', hue='algorithm', col='environment',
                data=selected_df, kind='line')
    # sns.relplot(x='epoch', y='success_rate', hue='algorithm',
    #             data=selected_df, kind='line')
    # sns.relplot(x='epoch', y='success_rate', hue='algorithm', data=selected_df,
    #             kind='line')
    pyplot.show()
    return df


if __name__ == '__main__':
    # draw_figure_from_data_frame(
    #     [{'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_2_singlenet_stop_l_0.1_rate_0.9',
    #       'algorithm': 'reward_stop'},
    #      {'dir': 'result/DPPO/DPPO_warm_1000_run_300_wm_2_singlenet',
    #       'algorithm': 'reward'},
    #      {'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_2',
    #       'algorithm': 'original'},
    #      {'dir': 'result/PPO/PPO_warm_1000_run_200',
    #       'algorithm': 'PPO'}])

    ppo_vs_ddq = [{'dir': 'result/dqn_p0_epoch1500',
                   'algorithm': 'DQN'},
                  {'dir': 'baseline/baseline_dqn_k5_5_agent_800_epoches',
                   'algorithm': 'DQN-5'},
                  {'dir': 'baseline/baseline_ddq_k5_5_agent_800_epoches',
                   'algorithm': 'DDQ-K5'},
                  {'dir': 'result/PPO/PPO_warm_1000_run_500_simulate_1024',
                   'algorithm': 'PPO'},
                  ]
    dppo_user_vs_world_model = [
        {'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet',
         'algorithm': 'DPPO-K5'},
        {'dir': 'result/PPO/PPO_warm_1000_run_500_simulate_1024',
         'algorithm': 'PPO'},
    ]
    dppo_early_stop = [{'dir': 'result/PPO/PPO_warm_1000_run_500_simulate_1024',
                        'algorithm': 'PPO'},
                       # {'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet',
                       #  'algorithm': 'DPPO-K5'},
                       {
                           'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet_stop_l_0.1_rate_0.9',
                           'algorithm': 'DPPO-K5-early-stop'},
                       ]

    test_world_model_net = [{'dir': 'result/DPPO/DPPO_run_40_wmnet_e100_plan_1_attention_layer_dropout',
                             'algorithm': 'attention_layer_dropout'},
                            # {'dir': 'result/DPPO/DPPO_warm_1000_run_200_wm_5_singlenet',
                            #  'algorithm': 'DPPO-K5'},
                            {
                                'dir': 'result/DPPO/DPPO_run_100_wmnet_original_e100_plan_1_again',
                                'algorithm': 'original'}, ]
    draw_figure_from_data_frame(test_world_model_net)
