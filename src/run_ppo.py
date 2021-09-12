# encoding:utf-8
"""
Created on May 22, 2016

This should be a simple minimalist run file. It's only responsibility should be to parse the arguments (which agent, user simulator to use) and launch a dialog simulation.

@author: xiul, t-zalipt, baolin
"""

import argparse, json, copy, os
import cPickle as pickle
import cPickle

from deep_dialog.dialog_system import text_to_dict, DialogManagerPPO
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, \
    RequestBasicsAgent, AgentPPO, Memory
from deep_dialog.usersims import RuleSimulator, ModelBasedSimulator
from deep_dialog.utils import init_logging, calculate_time

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

import numpy
import random
import os

import logging
import time
import sys
import torch
import config
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
seed = 5
numpy.random.seed(seed)
random.seed(seed)

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict_path', dest='dict_path', type=str,
                        default='./deep_dialog/data/dicts.v3.p',
                        help='path to the .json dictionary file')
    parser.add_argument('--movie_kb_path', dest='movie_kb_path', type=str,
                        default='./deep_dialog/data/movie_kb.1k.p',
                        help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str,
                        default='./deep_dialog/data/dia_acts.txt',
                        help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str,
                        default='./deep_dialog/data/slot_set.txt',
                        help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str,
                        default='./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p',
                        help='a list of user goals')
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='./deep_dialog/data/dia_act_nl_pairs.v6.json',
                        help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=40, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int,
                        help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.00, type=float,
                        help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int,
                        help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.00, type=float,
                        help='the intent err probability')

    # 9 为 DDQ 论文的模型
    parser.add_argument('--agt', dest='agt', default=9, type=int,
                        help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=1, type=int,
                        help='Select a user simulator. 0 is a Frozen user simulator.')

    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0,
                        help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
                        help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')

    parser.add_argument('--act_level', dest='act_level', type=int, default=0,
                        help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0,
                        help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0,
                        help='0 for no auto_suggest; 1 for auto_suggest')  # 只是增加打印数据库查询结果的信息
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0,
                        help='run_mode: 0 for NL; 1 for dia_act')

    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size',
                        type=int, default=5000,
                        help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=80,
                        help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False,
                        help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int,
                        default=50,
                        help='the size of validation set')
    # 0： 不预训练 world model。 1: 与 user 交互，预训练 world model。
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1,
                        help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100,
                        help='the number of epochs for warm start')
    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None,
                        help='the path for trained model')  # 如果已有模型，则不进行训练。如果没有模型，则进行训练
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str,
                        default='./deep_dialog/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10,
                        help='number of epochs for saving model')
    parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=100,
                        help='random seed for troch')

    # We changed to have a queue to hold experences. So this threshold will not be used to flush the buffer.
    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float,
                        default=0.6,
                        help='the threshold for success rate')

    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int,
                        help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str,
                        help='train/test/all; default is all')
    # ------------ 以上为 TC-Bot 自带的参数------------
    args = parser.parse_args()
    params = vars(args)  # 返回args的属性名与属性值构成的字典

init_logging(os.path.join(params['write_model_dir'], 'run.log'))
logging.debug('Dialog Parameters: ')
logging.debug(json.dumps(params, indent=4))

max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

dict_path = params['dict_path']
goal_file_path = params['goal_file_path']

# load the user goals from .p file
# 每个goal 包含 request_slots, diaact, inform_slots
all_goal_set = pickle.load(open(goal_file_path, 'rb'))

# split goal set
split_fold = params.get('split_fold', 5)
goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1:
        goal_set['test'].append(u_goal)
    else:
        goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

movie_kb_path = params['movie_kb_path']
movie_kb = pickle.load(open(movie_kb_path, 'rb'))

act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])

################################################################################
# a movie dictionary for user simulator - slot:possible values
################################################################################
movie_dictionary = pickle.load(open(dict_path, 'rb'))  # 所有slot可填的values

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']

agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['cmd_input_mode'] = params['cmd_input_mode']

# Manually set torch seed to ensure fail comparison.
torch.manual_seed(params['torch_seed'])

if agt == 0:
    agent = AgentCmd(movie_kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 4:
    agent = EchoAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 5:
    agent = RequestBasicsAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 9:
    agent = AgentPPO(movie_kb, act_set, slot_set, agent_params)

################################################################################
#    Add your agent here
################################################################################
else:
    pass

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']

if usr == 0:  # real user
    pass
    # user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1:
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)

################################################################################
#    Add your user simulator here
################################################################################
else:
    pass

################################################################################
# load trained NLG model
################################################################################
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)

################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)
################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManagerPPO(agent, user_sim, act_set, slot_set, movie_kb)

################################################################################
#   Run num_episodes Conversation Simulations
################################################################################

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']

""" Best Model and Performance Records """
# best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
best_res['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}
performance_records['dialog_number'] = {}

""" Save model """

""" save performance numbers """


def save_performance_records(path, filename, records):
    """
    保存 records 到 path/agt_{agt}_performance_records.json

    :param path: 目的目录
    :param agt: agent 的代号
    :param records:
    :param mode: string. train or test or evaluate
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"), indent=4)
        logging.info('saved performance in %s' % (filepath,))
    except Exception, e:
        logging.error('Error: Writing performance fails: %s' % (filepath,))
        logging.error(e)


""" Run N simulation Dialogues """


def simulation_epoch(total_time_steps=256, record_training_data=True):
    """
    simulation_epoch、simulation_epoch_for_training 只在 initialize_episode 时传入的参数有区别

    与 environment 交互 simulation_epoch_size 次。
    每次不为 world model 保存训练数据

    :param simulation_epoch_size: 执行多少个 episode
    :return: 交互的指标。 字典 result。包含keys={'success_rate', 'ave_reward', 'ave_turns'}
    """
    dialog_number = 0
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    # 模拟对话
    current_steps = 0
    while current_steps < total_time_steps:
        dialog_manager.initialize_episode()
        episode_over = False
        episode_reward = 0
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn(record_training_data=record_training_data)
            current_steps += 1

            # 记录信息
            if episode_over:
                cumulative_reward += episode_reward
                cumulative_turns += dialog_manager.state_tracker.turn_count
                dialog_number += 1
                if reward > 0:
                    successes += 1

    res = {}
    res['dialog_number'] = dialog_number
    res['success_rate'] = float(successes) / dialog_number
    res['ave_reward'] = float(cumulative_reward) / dialog_number
    res['ave_turns'] = float(cumulative_turns) / dialog_number
    logging.debug(
        "collect turn data {}, number of dialog {}, simulation success rate {}, ave reward {}, ave turns {}".format(
            current_steps, res['dialog_number'], res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


""" Warm_Start Simulation (by Rule Policy) """


@calculate_time
def warm_start_simulation():
    """
    用户与agent对话多轮，保存每轮的对话数据。所有轮次的对话结束后，用保存的对话数据训练 world model
    :return:
    """
    agent.use_rule = True
    memory = Memory()
    for i in range(10000):
        agent.ppo.memory.clear()
        dialog_manager.initialize_episode()
        while True:
            episode_over, reward = dialog_manager.next_turn(True)
            if episode_over:
                if reward > 0:
                    memory.append(agent.ppo.memory)
                break
    agent.ppo.memory = memory
    agent.imitate()
    agent.use_rule = False


@calculate_time
def run_episodes(count):
    '''

    :param count: number of episode
    :return
    '''
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # 最初时，world_model 与 agent 的 predict_mode == True
    if agt == 9 and params['trained_model_path'] == None and warm_start == 1:
        logging.info('warm_start starting ...')
        # 设置 agent.warm_start = 2，所以之后 agent 是否保存经验只由 agent.predict_mode 控制
        warm_start_simulation()
        logging.info('warm_start finished, start RL training ...')

    # 训练模型，记录结果
    if agt == 9 and params['trained_model_path'] == None:
        for episode in tqdm(xrange(count), desc=params['write_model_dir'].split('/')[-1],
                            mininterval=5):
            # agent.predict_mode = True
            simulation_res = simulation_epoch(1024, True)
            # agent.predict_mode = False
            agent.train()

            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']
            performance_records['dialog_number'][episode] = simulation_res['dialog_number']

            # 更新最好的指标
            if simulation_res['success_rate'] > best_res['success_rate']:
                # best_model['model'] = copy.deepcopy(agent)
                best_res['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['dialog_number'] = simulation_res['dialog_number']
                best_res['epoch'] = episode

            logging.debug(
                'best: epoch {}, success rate {}, average turns {}'.format(best_res['epoch'], best_res['success_rate'],
                                                                           best_res['ave_turns']))

            # save the model every 10 episodes
            if episode % save_check_point == 0 and params['trained_model_path'] == None:
                filename = 'epoch_{:04}_success_{}_ppo.pkl'.format(episode, best_res['success_rate'])
                filepath = os.path.join(params['write_model_dir'], filename)
                agent.save(filepath)

                save_performance_records(params['write_model_dir'], 'performance_epoch{}.json'.format(episode),
                                         performance_records)

        # 最后结果
        filepath = os.path.join(params['write_model_dir'], 'best_ppo.pkl')
        agent.save(filepath)
        save_performance_records(params['write_model_dir'], 'performance.json',
                                 performance_records)


run_episodes(200)
# run_episodes(num_episodes)
