# encoding:utf-8

import os
import argparse
import json
from matplotlib import pyplot


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
    return average_rate[:400]


if __name__ == '__main__':
    improve_1_dir = 'improve_td_err-newly_experience'
    improve_2_dir = 'improve-newly_experience'

    ddq = calculate_average_success_rate(result_dir='backup/baseline_ddq_k5_5_agent_800_epoches/',
                                         filename='train_performance.json')
    dqn = calculate_average_success_rate(result_dir='backup/baseline_dqn_k5_5_agent_800_epoches',
                                         filename='train_performance.json')
    result_1 = calculate_average_success_rate(
        result_dir=os.path.join('backup', improve_1_dir),
        filename='train_performance.json')
    result_2 = calculate_average_success_rate(
        result_dir=os.path.join('backup', improve_2_dir),
        filename='train_performance.json')

    # 画图
    epochs = list(range(1, len(ddq) + 1))
    pyplot.plot(epochs, ddq, label='DDQ 5')
    pyplot.plot(epochs, dqn, label='DQN 5')
    pyplot.plot(epochs, result_1, label=improve_1_dir)
    pyplot.plot(epochs, result_2, label=improve_2_dir)
    pyplot.xlabel('epochs')
    pyplot.ylabel('success rate')
    pyplot.legend()
    pyplot.show()
