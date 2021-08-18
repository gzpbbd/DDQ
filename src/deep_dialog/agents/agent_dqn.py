# encoding:utf-8

'''
Created on Oct 30, 2017

An DQN Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.

@author: Baolin Peng
'''

import logging
import random, copy, json
import cPickle as pickle
import numpy as np
from collections import namedtuple, deque

from deep_dialog import dialog_config

from agent import Agent
from deep_dialog.qlearning import DQN

import torch
import torch.optim as optim
import torch.nn.functional as F
import time

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        '''
        初始化参数，实例化DQN，并加载预训练的模型（如果指定了）
        :param movie_dict: 电影数据库，每个电影对应一个字典
        :param act_set: 可选的act集合
        :param slot_set: 可选的slot集合
        :param params: 配置参数
        '''
        self.movie_dict = movie_dict
        self.act_set = act_set  # dia_acts.txt得到的字典
        self.slot_set = slot_set  # slot_set.txt得到的字典
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        # deque 双端队列
        self.experience_replay_pool = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.running_expereince_pool = None  # hold experience from both user and world model

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        # predict mode 不需要保存对话数据。只根据状态输出结果。 而 train mode 需要保存对话数据入 replay pool，用于之后训练
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        # DQN 使用了numpy编写，只有两个dense层。最后一层输出为每个action对应的值，无激活函数
        self.device = torch.device(params['device'])
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(self.device)
        self.target_dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions).to(
            self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=1e-3)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.warm_start = 2

        # 自定义参数
        self.state_action_pairs = []
        self.policy_method = 'rl'
        self.policy_shaping_n = 1
        self.sampling_param_C = 0.8
        self.sampling_param_N = 300
        self.mode = 'train'

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0  # 对应了当前self.request_set请求的slot。只有在replay buffer满了之后起作用
        self.phase = 0  # 对应了当前的任务完成阶段。0: 任务未完成或者报告任务完成，1： 说thanks
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

        self.state_action_pairs.append({'state': self.representation, 'action': self.action})
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        '''
        使用one-hot或bag编码了state中的部分信息
        :param state: 当前状态的字典。键的集合为(agent_action, user_action, turn, current_slot,
        kb_results_dict, history)
        :return:
        '''
        """ Create the representation for each state """

        # print('---- agent state\n{}'.format(json.dumps(state,indent=4)))

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        #   Create one-hot of acts to represent the current user action
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        #     Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Creat bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Encode last agent act
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        #   Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        #   Encode last agent request slots
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        turn_rep = np.zeros((1, 1))

        #  One-hot representation of the turn count?
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # #   Representation of KB results (scaled counts)
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # #   Representation of KB results (binary)
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

        #   Representation of KB results (binary)
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep,
             agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        '''
        replay pool未满时使用rule_policy，否则使用DQN_policy
        :param representation:
        :return:
        '''
        """ epsilon-greedy policy """
        if random.random() < self.epsilon and self.mode == 'train':
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                return self.rule_policy()
            else:
                return self.DQN_policy(representation)

    def rule_policy(self):
        '''
        action 为先依次 request 几个固定的的槽位（self.request_set），然后inform taskcomplete、thanks
        :return: action对应的标号
        '''
        """ Rule Policy """

        act_slot_response = {}

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform",
                                 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def DQN_policy(self, state_representation):
        '''
        :param state_representation:
        :return: DQN网络预测出的最大概率的动作
       '''
        """ Return action from DQN"""

        state_representation = torch.FloatTensor(state_representation).to(self.device)

        with torch.no_grad():
            if self.policy_method == 'im':
                action = self.im_policy.predict(state_representation)
            elif self.policy_method == 'rl':
                action = self.dqn.predict(state_representation)
            elif self.policy_method == 'add':
                predict_im = self.im_policy.forward(state_representation)
                predict_im = F.softmax(predict_im, dim=-1)
                predict_rl = self.dqn.forward(state_representation)
                predict_rl = F.softmax(predict_rl, dim=-1)
                predict_result = torch.add(predict_im, predict_rl)
                action = torch.argmax(predict_result, dim=-1)
            elif self.policy_method == 'mul':
                predict_im = self.im_policy.forward(state_representation)
                predict_im = F.softmax(predict_im, dim=-1)
                predict_rl = self.dqn.forward(state_representation)
                predict_rl = F.softmax(predict_rl, dim=-1)
                predict_result = torch.mul(predict_im, predict_rl)
                action = torch.argmax(predict_result, dim=-1)
            elif self.policy_method == 'e':
                predict_im = self.im_policy.forward(state_representation)
                predict_im = F.softmax(predict_im, dim=-1)

                predict_rl = self.dqn.forward(state_representation)
                predict_rl = F.softmax(predict_rl, dim=-1)
                predict_rl = torch.exp(self.policy_shaping_n * predict_rl)

                predict_result = torch.mul(predict_im, predict_rl)
                action = torch.argmax(predict_result, dim=-1)
            elif self.policy_method == 'sample':
                predict_im = self.im_policy.forward(state_representation)
                predict_im = F.softmax(predict_im, dim=-1)

                predict_rl = self.dqn.forward(state_representation)
                predict_rl = F.softmax(predict_rl, dim=-1)

                predict_im = self.sample_and_recomputing_probability(predict_im)
                predict_result = torch.mul(predict_im, predict_rl)
                action = torch.argmax(predict_result, dim=-1)

        return action.cpu()

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user,
                                         from_model=False):
        """
        将当前轮的数据转为数字表示存入 replay_pool 中

        :param s_t:
        :param a_t:
        :param reward:
        :param s_tplus1:
        :param episode_over:
        :param st_user:
        :param from_model: 对话数据是否来自与 world model 的交互
        :return:
        """
        """ Register feedback from either environment or world model, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (
            state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if not from_model:
            self.experience_replay_pool.append(training_example)
        else:
            self.experience_replay_pool_from_model.append(training_example)

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type (int, ) -> Transition

        batch = random.sample(self.running_expereince_pool, batch_size)
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in xrange(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=100):
        """
        使用 experience buffer (user + world model) 训练 DQN 网络。

        :param batch_size:
        :param num_batches: 迭代的 epoch 的数量
        :return:
        """
        """ Train DQN with experience buffer that comes from both user and world model interaction."""

        self.cur_bellman_err = 0.
        self.cur_bellman_err_planning = 0.
        self.running_expereince_pool = list(self.experience_replay_pool) + list(
            self.experience_replay_pool_from_model)

        total_sample_time = 0
        total_net_time = 0

        for iter_batch in range(num_batches):
            for iter in range(len(self.running_expereince_pool) / (batch_size)):
                self.optimizer.zero_grad()
                batch = self.sample_from_buffer(batch_size)

                # to device
                state = torch.FloatTensor(batch.state).to(self.device)
                action = torch.tensor(batch.action).to(self.device)
                reward = torch.FloatTensor(batch.reward).to(self.device)
                next_state = torch.FloatTensor(batch.next_state).to(self.device)
                term = np.asarray(batch.term, dtype=np.float32)
                term = torch.FloatTensor(term).to(self.device)

                # 计算 Q 值
                state_value = self.dqn(state).gather(1, action)
                next_state_value, _ = self.target_dqn(next_state).max(1)
                next_state_value = next_state_value.unsqueeze(1)
                expected_value = reward + self.gamma * next_state_value * (
                        1 - term)

                loss = F.mse_loss(state_value, expected_value)
                loss.backward()
                self.optimizer.step()
                self.cur_bellman_err += loss.item()

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            # print 'saved model in %s' % (path,)
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path,)
            print e

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print "Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    # def set_user_planning(self, user_planning):
    #     '''
    #
    #     :param user_planning: world model
    #     :return:
    #     '''
    #     self.user_planning = user_planning

    def save(self, filename):
        torch.save(self.dqn.state_dict(), filename)
        logging.info('saved dqn model to {}'.format(filename))

    def load(self, filename):
        self.dqn.load_state_dict(torch.load(filename))
        logging.info('loaded dqn model from {}'.format(filename))

    def reset_dqn_target(self):
        """
        更新 target DQN 的参数

        :return:
        """
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def empty_s_a_pairs(self):
        self.state_action_pairs = []

    def get_s_a_pairs(self):
        return self.state_action_pairs

    # 自定义函数
    def add_im_policy(self, im_policy):
        self.im_policy = im_policy

    def set_policy_method(self, method):
        self.policy_method = method

    def set_exp_param_n(self, n):
        self.policy_shaping_n = n

    def sample_and_recomputing_probability(self, probability):
        probability_len = probability.shape[-1]

        # 采样，计算每种样本的出现次数
        samples = torch.multinomial(probability, self.sampling_param_N, replacement=True)
        occurrences = torch.stack(
            [torch.bincount(row, minlength=probability_len) for row in samples], dim=0)

        # 按照论文公式 （4） 调整出现次数
        adjust_occurrences = (
                occurrences - self.sampling_param_N * (1 - self.sampling_param_C)).float()

        # 按照论文公式 （3） 计算出现调整后的概率分布
        c_temporary = self.sampling_param_C ** adjust_occurrences
        rest_c_temporary = (1 - self.sampling_param_C) ** adjust_occurrences
        adjust_probability = c_temporary / (c_temporary + rest_c_temporary)
        return adjust_probability

    def set_sampling_param_c(self, c):
        self.sampling_param_C = c

    def set_sampling_param_n(self, n):
        self.sampling_param_N = n

    def set_train_mode(self):
        self.mode = 'train'

    def set_evaluate_mode(self):
        self.mode = 'evaluate'
