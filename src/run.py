# encoding:utf-8
"""
Created on May 22, 2016

This should be a simple minimalist run file. It's only responsibility should be to parse the arguments (which agent, user simulator to use) and launch a dialog simulation.

@author: xiul, t-zalipt, baolin
"""
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

from tqdm import tqdm
import argparse, json, copy, os
import cPickle as pickle
from collections import OrderedDict
import time

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, \
    RequestBasicsAgent, AgentDQN
from deep_dialog.usersims import RuleSimulator, ModelBasedSimulator

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *
import logging
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

import numpy
import random
import os
from deep_dialog.imitation_model import ImitationPolicy

seed = 5
numpy.random.seed(seed)
random.seed(seed)

import torch
import numpy as np

start_time = time.time()

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""


def init_logging(filepath):
    # file handler
    abs_path = os.path.abspath(filepath)
    dir = os.path.dirname(abs_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_handler = logging.FileHandler(filepath, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(message)s\n                       - %(levelname)s %(asctime)s %(filename)s(%(funcName)s %(lineno)d)'))

    # stream handle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s: %(message)s",
                                                   datefmt='%H:%M:%S'))

    # setting logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)


def parset_params():
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
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=3,
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
    # 0： 不预训练 world model。 1: 与 user 交互，预训练 world model。2 预训练 world model 结束
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1,
                        help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100,
                        help='the number of epochs for warm start')
    parser.add_argument('--planning_steps', dest='planning_steps', type=int, default=4,
                        help='the number of planning steps with world model')

    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None,
                        help='the path for trained model')  # 如果已有DDQ模型，则不进行训练。如果没有模型，则进行训练
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str,
                        default='./deep_dialog/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10,
                        help='number of epochs for saving model')

    # We changed to have a queue to hold experences. So this threshold will not be used to flush the buffer.
    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float,
                        default=0.6,
                        help='the threshold for success rate')

    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int,
                        help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str,
                        help='train/test/all; default is all')
    # have changed the name
    parser.add_argument('--K_DQN', dest='K_DQN', type=int, default=0,
                        help='planning k steps with environment rather than world model')
    parser.add_argument('--boosted', dest='boosted', type=int, default=1,
                        help='Boost the world model')
    parser.add_argument('--train_world_model', dest='train_world_model', type=int, default=1,
                        help='Whether train world model on the fly or not')
    parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=100,
                        help='random seed for troch')

    # 自定义参数
    parser.add_argument('--policy_method', dest='policy_method', type=str, default='rl',
                        help='the method of policy(rl, im, sample, e, add, mul)')
    parser.add_argument('--validation_epoch_size', dest='validation_epoch_size', type=int,
                        default=300,
                        help='the size of validation set')
    parser.add_argument('--validation_span', type=int,
                        default=10,
                        help='after how many epochs validating one time')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='using which device to run experiment')
    parser.add_argument('--to_do_what', type=str,
                        default='train_rl',
                        help='train_rl or train_im')  # 运行模式：train_rl 训练DDQ; train_im 利用DDQ模型训练IM模型。
    parser.add_argument('--conversations_number_for_im_model', type=int,
                        default=100000,
                        help='how many conversations agent play with environment for training im '
                             'model')

    args = parser.parse_args()
    params = vars(args)  # 返回args的属性名与属性值构成的字典

    # print 'Dialog Parameters: '
    # print json.dumps(params, indent=2)
    return params


params = parset_params()

init_logging(os.path.join(params['write_model_dir'], 'run.log'))
logging.info('Dialog Parameters: ')
logging.info(json.dumps(params, indent=2))

if params['device'].startswith('cuda'):
    os.environ["CUDA_VISIBLE_DEVICES"] = params['device'].split(':', 1)[-1]
    params['device'] = 'cuda'

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
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']
agent_params['device'] = params['device']

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
    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)

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
usersim_params['hidden_size'] = params['dqn_hidden_size']

if usr == 0:  # real user
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1:
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    # agent.set_user_planning(world_model)
# elif usr == 2:
#     user_sim = ModelBasedSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)

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
world_model.set_nlg_model(nlg_model)

################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)
world_model.set_nlu_model(nlu_model)

################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManager(agent, user_sim, world_model, act_set, slot_set, movie_kb)

################################################################################
#   Run num_episodes Conversation Simulations
################################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

# simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']
planning_steps = params['planning_steps']

agent.planning_steps = planning_steps

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']

""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = OrderedDict()
performance_records['success_rate'] = OrderedDict()
performance_records['ave_turns'] = OrderedDict()
performance_records['ave_reward'] = OrderedDict()

""" Save model """


def save_model(path, agent_model, filename='best_model.pkl'):
    """
    保存模型到 path/agt_{agt}_{best_epoch}_{cur_epoch}_{success_rate}.pkl

    :param path: 目的目录
    :param agent_model: agent 模型
    :param filename: 文件名
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    try:
        agent_model.save(filepath)
        logging.info('save agent model to {}'.format(filepath))
    except Exception, e:
        logging.info('Error: Writing model fails: %s' % (filepath,))
        logging.info(e)

    return filepath


""" save performance numbers """


def save_performance_records(path, records, filename='performance.json'):
    """
    保存 records 到 path/agt_{agt}_performance_records.json

    :param path: 目的目录
    :param records: 字典格式的数据
    :param filename: 文件名
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"), indent=4)
        logging.info('saved performance record in %s' % (filepath,))
    except Exception, e:
        logging.info('Error: Writing model fails: %s' % (filepath,))
        logging.info(e)


""" Run N simulation Dialogues """


def computer_metrics(simulation_epoch_size):
    """
    computer_metrics、simulation_epoch_for_training 只在 initialize episode 时传入的参数有区别

    与环境交互，计算评价指标
    agent 与 environment 交互 simulation_epoch_size 次。
    每次不为 world model 保存训练数据

    :param simulation_epoch_size: 执行多少个 episode
    :return: 交互的指标。 字典 result。包含keys={'success_rate', 'ave_reward', 'ave_turns'}
    """
    agent.set_evaluate_mode()

    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in tqdm(xrange(simulation_epoch_size), desc='computer_metrics'):
        dialog_manager.initialize_episode(use_environment=True)
        episode_over = False

        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=False,
                                                            record_training_data_for_user=False)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                cumulative_turns += dialog_manager.state_tracker.turn_count

    agent.set_train_mode()

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    return res


def simulation_epoch_for_training(simulation_epoch_size, K_DQN=False):
    """
    computer_metrics、simulation_epoch_for_training 只在 initialize episode 时传入的参数有区别

    执行 simulation_epoch_size 个 episode 的对话。第一个 episode 使用 environment 与 system 交互。
    其余 episode 使用 world model 与 system 交互。
    每次为 world model 保存训练数据

    :param simulation_epoch_size: 执行多少个 episode
    :param K_DQN: true- 全部使用 user simulator。false- DDQ。既只有第一轮使用 user simulator，其他轮使用 world model
    :return: 交互的指标。 字典 result。包含keys={'success_rate', 'ave_reward', 'ave_turns'}
    """
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    # res = {}
    for episode in xrange(simulation_epoch_size):

        # 控制是否使用 world model
        if K_DQN or episode % simulation_epoch_size == 0:
            use_environment = True
        else:
            use_environment = False

        # 做DDQ实验时，K_DQN 为 0，所以可以忽略它。当做 K-DQN 实验时， K_DQN 为 1.表示基线
        dialog_manager.initialize_episode(use_environment)

        episode_over = False
        while not episode_over:
            episode_over, reward = dialog_manager.next_turn(record_training_data=True,
                                                            record_training_data_for_user=True)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                cumulative_turns += dialog_manager.state_tracker.turn_count


""" Warm_Start Simulation (by Rule Policy) """


def warm_start_simulation():
    """
    用户与agent对话，保存对话数据。训练 world model

    :return:
    """
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for episode in xrange(warm_start_epochs):
        dialog_manager.initialize_episode(use_environment=True)  # 初始化目标，第一个用户动作，更新DST状态
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                cumulative_turns += dialog_manager.state_tracker.turn_count

        warm_start_run_epochs += 1

        # if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
        #     break

    if params['boosted']:
        world_model.train(batch_size, num_batches=5)
    else:
        world_model.training_examples = []

    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_epochs
    logging.info("warm starting done. %s epochs, success rate %s, ave reward %s, ave turns %s" % (
        episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    logging.info(
        "    Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))


def run_episodes(count):
    '''

    :param count: number of episode
    :param status: 
    :return:
    '''
    K_DQN = params['K_DQN']
    simulation_epoch_size = planning_steps + 1

    if agt == 9 and params['trained_model_path'] is None and warm_start == 1:
        logging.info('warm start starting ...')
        warm_start_simulation()

    for episode in xrange(count):
        logging.info("Episode: %s" % episode)
        # print 'experience_replay_pool {}, experience_replay_pool_from_model {}'.format(
        #     len(agent.experience_replay_pool), len(agent.experience_replay_pool_from_model))

        # simulation
        if agt == 9 and params['trained_model_path'] is None:
            # 运行 simulation_epoch_size 轮，第一轮使用 user，之后都是使用 world model，保存数据
            simulation_epoch_for_training(simulation_epoch_size, K_DQN)
            agent.train(batch_size, 1)
            agent.reset_dqn_target()
            if params['train_world_model']:
                world_model.train(batch_size, 1)

            # 每隔 10 次计算一次精度，最后一轮也会计算精度
            if episode % params['validation_span'] != 0 and episode != count - 1:
                continue

            # 运行 50 轮，全部使用 user，不使用 world model，计算指标
            simulation_res = computer_metrics(params['validation_epoch_size'])
            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']
            # 更新最好的指标
            if simulation_res['success_rate'] > best_res['success_rate']:
                best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode

            logging.info(
                "Running - Simulation success rate %s, Ave reward %s, Ave turns %s, "
                "Best success "
                "rate %s" % (
                    performance_records['success_rate'][episode],
                    performance_records['ave_reward'][episode],
                    performance_records['ave_turns'][episode], best_res['success_rate']))

    if agt == 9 and params['trained_model_path'] is None:
        save_model(params['write_model_dir'], best_model['model'])
        save_performance_records(params['write_model_dir'], performance_records)


def using_agent_play_conversation_with_user(
        agent_path='baseline/baseline_ddq_k5_5_agent_800_epoches/run2/agt_best.pkl'
        , s_a_path=None, epochs=100):
    # 加载 rl 模型，与 environment 交互，基于所有完成用户任务的对话，生成 state-action 数据集，保存到 s_a_path 中
    agent.warm_start = 2  # 设置为2时，不使用 agent 的 rule policy
    agent.load(agent_path)
    state_action_pairs = []

    succeed_amount = 0
    for episode in tqdm(xrange(epochs)):
        agent.empty_s_a_pairs()
        dialog_manager.initialize_episode(use_environment=True)
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=False,
                                                            record_training_data_for_user=False)

            if episode_over and reward > 0:
                state_action_pairs.extend(agent.get_s_a_pairs())
                succeed_amount += 1

    state = [torch.tensor(pair['state']) for pair in state_action_pairs]
    action = [torch.tensor(pair['action']) for pair in state_action_pairs]
    state = torch.stack(state, dim=0).squeeze(dim=1)
    action = torch.stack(action, dim=0).squeeze(dim=1)
    if s_a_path:
        pickle.dump({'state': state, 'action': action}, open(s_a_path, 'wb'))
    logging.info(
        'generating state-action pairs: state_action_pairs {}, succeed_amount {}'.format(
            len(state_action_pairs), succeed_amount))


if params['to_do_what'] == 'train_rl':
    # 训练 rl 模型
    run_episodes(num_episodes)
elif params['to_do_what'] == 'generate_state_action_pairs':
    # 加载 rl 模型，与 environment 交互，生成 state-action 数据集
    rl_model_path = os.path.join(params['write_model_dir'], 'best_model.pkl')
    s_a_path = os.path.join(params['write_model_dir'], 's_a_pairs.pkl')
    using_agent_play_conversation_with_user(rl_model_path, s_a_path,
                                            epochs=params['conversations_number_for_im_model'])
elif params['to_do_what'] == 'train_im':
    # 基于 state-action 数据集，训练 im 模型
    # rl_model_path = os.path.join(params['write_model_dir'], 'best_model.pkl')
    im_policy = ImitationPolicy(agent.state_dimension, agent.num_actions, params['device'])

    s_a_path = os.path.join(params['write_model_dir'], 's_a_pairs.pkl')
    im_policy.create_data_loader(s_a_path)

    im_policy.train(3)
    im_model_path = os.path.join(params['write_model_dir'], 'im_model.pkl')
    im_policy.save(im_model_path)

    # 将 im 模型注入 agent 中，测试 im 模型对话成功率
    agent.add_im_policy(im_policy)
    agent.set_policy_method('im')
    agent.warm_start = 2
    simulation_res = computer_metrics(params['validation_epoch_size'])
    logging.info('im model validate: {}'.format(simulation_res))

logging.info('total time: {:.1f} minutes'.format((time.time() - start_time) / 60))

# dqn_model_path = 'baseline/baseline_dqn_k5_5_agent_800_epoches/run4/agt_best.pkl'
# s_a_path = 'result/ddq_s_a_path.pkl'  # 'result/ddq_s_a_path.pkl'  #
# 'result/dqn_4_s_a_path.pkl'
# im_model_path = 'result/dqn_4_im_model_100000.pkl'

# 生成对话数据并保存
# using_agent_play_conversation_with_user(dqn_model_path, s_a_path, epochs=100)

# 训练 im 模型
# im_policy = ImitationPolicy(agent.state_dimension, agent.num_actions, params['device'])
# im_policy.create_data_loader(s_a_path)
# im_policy.train(3)
# im_policy.save(im_model_path)

# 加载 im 模型
# im_policy.load(im_model_path)
# agent.add_im_policy(im_policy)
# # 加载 rl 模型
# agent.load(dqn_model_path)
#
# agent.warm_start = 2
# 测试
# for method in ['e']:
#     for n in [0, 0.05, 0.1, 0.3, 1, 3, 9]:
#         agent.set_policy_method(method)
#         agent.set_shaping_param(n)
#         simulation_res = computer_metrics(params['validation_epoch_size'])
#         print method, n, 'policy', simulation_res

#
# agent.evaluate()
# print 'evaluate mode'
# for method in ['im', 'rl']:
#     agent.set_policy_method(method)
#     simulation_res = computer_metrics(params['validation_epoch_size'])
#     print method, 'policy', simulation_res
#
# agent.train()
# print 'train mode'
# for method in ['im', 'rl']:
#     agent.set_policy_method(method)
#     simulation_res = computer_metrics(params['validation_epoch_size'])
#     print method, 'policy', simulation_res

#
# agent.set_policy_method('sample')
# for c in [0.8]:
#     for n in [300]:
#         agent.set_sampling_param_c(c)
#         agent.set_sampling_param_n(n)
#         simulation_res = computer_metrics(params['validation_epoch_size'])
#         print 'sampling C {}, N {} policy {}'.format(c, n, simulation_res)


# agent.set_policy_method(params['policy_method'])
# agent.set_sampling_param_c(0.8)
# agent.set_sampling_param_n(300)
