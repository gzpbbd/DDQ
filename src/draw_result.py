# encoding:utf-8

import os
import argparse
import json
from matplotlib import pyplot
import pandas as pd
import seaborn as sns


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


def read_to_data_frame(result_dir, algorithm, filename):
    df = pd.DataFrame(columns=['algorithm', 'run_num', 'epoch', 'success_rate'])
    sub_dir_list = os.listdir(result_dir)
    for sub_dir in sub_dir_list:  # 对每个子目录的文件
        # sub_dir_path = os.path.join(result_dir, sub_dir)
        file_path = os.path.join(result_dir, sub_dir, filename)
        with open(file_path, 'r') as f:
            result = json.load(f)  # 读取记录
            for epoch, success_rate in result['success_rate'].items():
                df = df.append([{'algorithm': algorithm, 'run_num': sub_dir, 'epoch': int(epoch),
                                 'success_rate': success_rate}])  # 加入 data frame 中
    return df


def draw_figure_from_data_frame(result_dirs, filename='performance.json'):
    all_df = []
    for dir in result_dirs:
        result_df = read_to_data_frame(dir['dir'], dir['algorithm'], filename)
        all_df.append(result_df)
        print len(result_df)
    df = pd.concat(all_df)
    df.sort_values(by='run_num', inplace=True)
    sns.relplot(x='epoch', y='success_rate', hue='algorithm', data=df[df['epoch'] < 200],
                kind='line')
    pyplot.show()
    return df


if __name__ == '__main__':
    draw_figure_from_data_frame(
        [{'dir': 'result/PPO', 'algorithm': 'PPO'}, {'dir': 'result/DPPO', 'algorithm': 'DPPO'}])
