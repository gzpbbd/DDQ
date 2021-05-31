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
    return average_rate


if __name__ == '__main__':
    ddq_suc_rate = calculate_average_success_rate(result_dir='backup/result_improved_replay_pool--10_agent--500_epoches/ddq',
                                                  filename='train_performance.json')
    improved_suc_rate = calculate_average_success_rate(result_dir='backup/result_improved_replay_pool--10_agent--500_epoches/improve',
                                                       filename='train_performance.json')

    # 画图
    epochs = list(range(1, len(ddq_suc_rate) + 1))
    pyplot.plot(epochs, ddq_suc_rate, label='DDQ')
    pyplot.plot(epochs, improved_suc_rate, label='improved_DDQ')
    pyplot.xlabel('epochs')
    pyplot.ylabel('success rate')
    pyplot.legend()
    pyplot.show()
