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


dirs = [
    # attention_r_cat different epochs
    # {'dir': 'result/DPPO/DPPO_run_40_wmnet_attention_r_cat_sigmoid_e100_plan_1',
    #  'algorithm': 'attention_r_cat_train_100'},
    # # attention_r different epochs
    # {'dir': 'result/DPPO/DPPO_run_40_wmnet_attention_r_e30_plan_1',
    #  'algorithm': 'attention_r_train_30'},
    # {'dir': 'result/DPPO/DPPO_run_40_wmnet_attention_r_sigmoid_e100_plan_1',
    #  'algorithm': 'attention_r_train_100'},
    # original different wm train epochs
    {'dir': 'result/DPPO/DPPO_run_100_wmnet_original_more_fc_e100_plan_1',
     'algorithm': 'original_train_100_more_fc'},
    {'dir': 'result/DPPO/DPPO_run_100_wmnet_original_e100_plan_1',
     'algorithm': 'original_train_100'},
    {'dir': 'result/DPPO/DPPO_run_100_wmnet_original_e100_plan_1_again',
     'algorithm': 'original_train_100_again'},
]
df = read_data(dirs)
df = df[df['epoch'] < 40]
sns.relplot(x='epoch', y='acc', hue='algorithm', col='acc_type', data=df, kind='line')
plt.show()
