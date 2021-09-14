# encoding:utf-8
from .usersim import UserSimulator
import argparse, json, random, copy, sys
import numpy as np
from user_model import SimulatorModel
from collections import namedtuple, deque
from deep_dialog import dialog_config

import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import time
import config
from torch import nn
import math

WorldModelTransition = namedtuple('Transition',
                                  ('state', 'agent_action', 'reward', 'term', 'user_action'))


class WorldModelSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        '''
        :param movie_dict: 字典。电影领域每个slot的候选value
        :param act_set: 可选的act集合
        :param slot_set: 可选的slot集合
        :param start_set: 第一轮的goal集合
        :param params: 配置参数
        '''
        """ Constructor shared by all user simulators """

        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.feasible_actions_users = dialog_config.feasible_actions_users
        self.num_actions_sys = len(self.feasible_actions)
        self.num_actions_user = len(self.feasible_actions_users)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 9 * self.slot_cardinality + 3 + self.max_turn

        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        self.learning_phase = params['learning_phase']

        # SimulatorModel 模型结构对应了论�? world model 的描�?
        self.model = WorldModelNet(self.num_actions_sys, self.state_dimension,
                                   self.num_actions_user, 1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)
        # 关于DPPO
        self.memory = WorldModelMemory()

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(dialog_config.start_dia_acts.keys())

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(self.goal['inform_slots'].keys())
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal[
                'inform_slots'].keys():  # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        return self.sample_goal

    def prepare_user_goal_representation(self, user_goal):
        """"""

        request_slots_rep = np.zeros((1, self.slot_cardinality))
        inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for s in user_goal['request_slots']:
            s = s.strip()
            request_slots_rep[0, self.slot_set[s]] = 1
        for s in user_goal['inform_slots']:
            s = s.strip()
            inform_slots_rep[0, self.slot_set[s]] = 1
        self.user_goal_representation = np.hstack([request_slots_rep, inform_slots_rep])

        return self.user_goal_representation

    def train(self, epochs=10):
        dataset = self.memory.get_batch()
        data_size = len(self.memory)
        logging.debug('World Model: train on {} samples'.format(data_size))
        state = torch.from_numpy(np.stack(dataset.state)).float()
        agent_action = torch.from_numpy(np.stack(dataset.agent_action))
        reward = torch.from_numpy(np.stack(dataset.reward)).float().unsqueeze(-1)
        term = torch.from_numpy(np.stack(dataset.term).astype(int)).float().unsqueeze(-1)
        user_action = torch.from_numpy(np.stack(dataset.user_action))

        span = epochs // 5
        for epoch in range(epochs):
            perm = torch.randperm(data_size)
            state_shuf, agent_action_shuf, reward_shuf, term_shuf, user_action_shuf = state[perm], \
                                                                                      agent_action[
                                                                                          perm], \
                                                                                      reward[perm], \
                                                                                      term[perm], \
                                                                                      user_action[
                                                                                          perm]
            optim_chunk_num = int(math.ceil(data_size * 1.0 / 64))
            state_shuf = torch.chunk(state_shuf, optim_chunk_num)
            agent_action_shuf = torch.chunk(agent_action_shuf, optim_chunk_num)
            reward_shuf = torch.chunk(reward_shuf, optim_chunk_num)
            term_shuf = torch.chunk(term_shuf, optim_chunk_num)
            user_action_shuf = torch.chunk(user_action_shuf, optim_chunk_num)

            total_loss_r = 0.
            total_loss_t = 0.
            total_loss_u_a = 0.
            for state_b, agent_action_b, reward_b, term_b, user_action_b in zip(state_shuf,
                                                                                agent_action_shuf,
                                                                                reward_shuf,
                                                                                term_shuf,
                                                                                user_action_shuf):
                self.optimizer.zero_grad()
                reward_, term_, user_action_ = self.model(state_b, agent_action_b)
                loss_r = F.mse_loss(reward_, reward_b)
                loss_t = F.binary_cross_entropy_with_logits(term_, term_b)
                loss_u_a = F.nll_loss(user_action_, user_action_b)
                loss = loss_r + loss_t + loss_u_a
                loss.backward()
                self.optimizer.step()

                total_loss_r += loss_r.item() / optim_chunk_num
                total_loss_t += loss_t.item() / optim_chunk_num
                total_loss_u_a += loss_u_a.item() / optim_chunk_num

            if epoch % span == 0 or epoch == epochs - 1:
                acc_reward, acc_term, acc_user_action = self.calculate_metrics(state, agent_action,
                                                                               reward, term,
                                                                               user_action)

                logging.debug('Training World Model: iteration {}/{}'.format(epoch, epochs))
                logging.debug(
                    '                      loss: reward {:.05f}, term {:.05f}, user action {:.05f}'.format(
                        total_loss_r, total_loss_t, total_loss_u_a))

                logging.debug(
                    '                      acc_reward {:.03f}, acc_term {:.03f}, acc_user_action {:.03f}'.format(
                        acc_reward, acc_term, acc_user_action))
            # logging.debug('training World Model: iteration {}'.format(i))
            # logging.debug('       acc_reward      {:.03f}, reward loss {}'.format(acc_reward, total_loss_r))
            # logging.debug('       acc_term        {:.03f}, term loss {}'.format(acc_term, total_loss_t))
            # logging.debug('       acc_user_action {:.03f}, user_action loss {}'.format(acc_user_action, total_loss_u_a))

    def calculate_metrics(self, state, agent_action, reward, term, user_action):
        reward_pred, term_pred, user_action_pred = self.model.predict(state, agent_action)
        # for reward
        reward_pred_ = torch.full(reward_pred.shape, -0.1)
        reward_pred_[reward_pred > 1] = 1
        reward_pred_[reward_pred < -1] = -1
        acc_reward = 1.0 * torch.sum(reward_pred_ == reward).item() / len(reward.view(-1))
        # for term
        term = term.int()
        term_pred = term_pred > 0.5
        acc_term = 1.0 * torch.sum(term == term_pred.int()).item() / len(term.view(-1))
        # for user_action
        acc_user_action = 1.0 * torch.sum(user_action == user_action_pred).item() / len(
            term.view(-1))
        return acc_reward, acc_term, acc_user_action

    def save_experience(self, user_state, agent_action, reward, term, user_action):
        user_state_vec = self.prepare_state_representation(user_state)
        goal_vec = self.prepare_user_goal_representation(self.sample_goal)
        state_vec = np.hstack([user_state_vec, goal_vec]).squeeze()
        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1
        elif reward == -1:
            reward = -0.1
        action_index = self.action_index(copy.deepcopy(user_action))
        self.memory.push(state_vec, agent_action, reward, term, action_index)

    def next(self, s, a):
        """
        Provide
        :param s: state representation from tracker
        :param a: last action from agent
        :return: next user action, termination and reward predicted by world model
        """

        self.state['turn'] += 2
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn):
            reward = - self.max_turn
            term = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            return response_action, term, reward

        s = self.prepare_state_representation(s)
        g = self.prepare_user_goal_representation(self.sample_goal)
        s = np.hstack([s, g])
        reward, term, action = self.predict(torch.from_numpy(s).float().squeeze(),
                                            torch.from_numpy(np.asarray(a)))
        action = action.item()
        reward = reward.item()
        term = term.item()
        action = copy.deepcopy(self.feasible_actions_users[action])

        if action['diaact'] == 'inform':
            if len(action['inform_slots'].keys()) > 0:
                slots = action['inform_slots'].keys()[0]
                if slots in self.sample_goal['inform_slots'].keys():
                    action['inform_slots'][slots] = self.sample_goal['inform_slots'][slots]
                else:
                    action['inform_slots'][slots] = dialog_config.I_DO_NOT_CARE

        response_action = action

        term = term > 0.5

        if reward > 1:
            reward = 2 * self.max_turn
        elif reward < -1:
            reward = -self.max_turn
        else:
            reward = -1

        return response_action, term, reward

    def predict(self, s, a):
        return self.model.predict(s, a)

    def register_user_goal(self, goal):
        self.user_goal = goal

    def action_index(self, act_slot_response):
        """ Return the index of action """
        del act_slot_response['turn']
        del act_slot_response['nl']

        for i in act_slot_response['inform_slots'].keys():
            act_slot_response['inform_slots'][i] = 'PLACEHOLDER'

        # rule
        if act_slot_response['diaact'] == 'request': act_slot_response['inform_slots'] = {}
        if act_slot_response['diaact'] in ['thanks', 'deny', 'closing']:
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {}

        for (i, action) in enumerate(self.feasible_actions_users):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1, 1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))
        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
        #     kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep,
             agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep])
        return self.final_representation


class WorldModelMemory:
    def __init__(self):

        self.memory = []
        self.max_size = 3000

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(WorldModelTransition(*args))
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return WorldModelTransition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return WorldModelTransition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class WorldModelNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=60):
        super(WorldModelNet, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)

    def forward(self, s, a):
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze(1)
        # h = F.tanh(h_s + h_a)
        h = torch.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), 1)

        return reward, term, action

    def predict(self, s, a):
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a)
        h = torch.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = torch.sigmoid(self.linear_h2t(h))
        action = F.log_softmax(self.linear_h2a(h), -1)

        return reward, term, action.argmax(-1)
