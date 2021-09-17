# encoding:utf-8

import os
import argparse
import json
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from deep_dialog.utils import SmoothValue
from collections import OrderedDict


def read_data(dir_list):
    dfs = []
    filename = 'world_model_train_result.csv'
    for dir_algorithm in dir_list:
        for sub_dir in os.listdir(dir_algorithm['dir']):
            filepath = os.path.join(dir_algorithm['dir'], sub_dir, filename)
            print os.path.abspath(filepath)
            df = pd.read_csv(filepath)
            # df.insert(0, 'algorithm', )
            reward_df = pd.DataFrame(
                {'epoch': df['epoch'], 'acc': df['acc_reward'], 'acc_type': 'reward',
                 'algorithm': dir_algorithm['algorithm']})
            term_df = pd.DataFrame({'epoch': df['epoch'], 'acc': df['acc_term'], 'acc_type': 'term',
                                    'algorithm': dir_algorithm['algorithm']})
            user_action_df = pd.DataFrame(
                {'epoch': df['epoch'], 'acc': df['acc_user_action'], 'acc_type': 'user_action',
                 'algorithm': dir_algorithm['algorithm']})

            dfs.append(pd.concat([reward_df, term_df, user_action_df]))
    dfs = pd.concat(dfs)
    return dfs


dirs = [{'dir': 'result/_DPPO/wm_dataset/attention_on_r_epoch_100',
         'algorithm': 'attention_r'},

        {
            'dir': 'result/_DPPO/wm_dataset/original_100_epoch',
            'algorithm': 'original'}, ]
df = read_data(dirs)
sns.relplot(x='epoch', y='acc', hue='algorithm', col='acc_type', data=df, kind='line')
plt.show()
