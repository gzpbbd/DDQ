# encoding:utf-8

'''
Created on Oct 30, 2017

An DQN Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.

@author: Baolin Peng
'''

import random, copy, json
import cPickle as pickle
import numpy as np
from collections import namedtuple, deque

from deep_dialog import dialog_config
import math

from agent import Agent
from deep_dialog.qlearning import DQN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import time
import config

PPOTransition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentPPO(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)

        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.ppo = PPO(self.state_dimension, self.num_actions)  # todo

        self.cur_bellman_err = 0
        self.use_rule = False

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

    def state_to_action(self, state):  # todo
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        '''

        :param state: 字典类型
        :return: numpy 中的 一维向量
        '''
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

        turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
            kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def run_policy(self, representation):
        '''

        :param representation: 一维向量
        :return: int
        '''
        if self.use_rule:
            return self.rule_policy()
        else:
            return self.ppo.predict(representation)  # todo

    def rule_policy(self):
        """ Rule Policy """

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None

    def save_experience(self, s, a, r, episode_over):  # todo
        """ Register feedback from the environment, to be stored as future training data """
        s_vec = self.prepare_state_representation(s).squeeze()
        a_int = self.action
        a_vec = np.array(a_int)
        self.ppo.save_experience(s_vec, a_vec, r, episode_over)

    def train(self):  # todo
        """ Train DQN with experience replay """
        self.ppo.train()

    def imitate(self):
        self.ppo.imitate()

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):  # todo
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path,)
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path,)
            print e

    def load_experience_replay_from_file(self, path):  # todo
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):  # todo
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        print "trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    def save(self, filename):
        self.ppo.save(filename)
        logging.debug('saved PPO model to {}'.format(filename))

    def load(self, filename):
        self.ppo.load(filename)
        logging.debug('loaded PPO model to {}'.format(filename))


class PPO:
    def __init__(self, num_inputs, num_outputs):
        self.actor = DiscreteActor(num_inputs, num_outputs)
        self.critic = Critic(num_inputs)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=0.0001)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.00005)

        self.memory = PPOMemory()
        # PPO的参数
        self.gamma = 0.99
        self.tau = 0.95
        self.epsilon = 0.2

    def save_experience(self, s_vec, a_vec, r, episode_over):
        mask = 0 if episode_over else 1
        self.memory.push(s_vec, a_vec, r, mask)

    def train(self):
        batch = self.memory.get_batch()
        batch_size = len(self.memory)
        self.memory.clear()
        logging.debug('PPO: train on {} samples'.format(batch_size))

        s = torch.from_numpy(np.stack(batch.state)).float()
        a = torch.from_numpy(np.stack(batch.action))
        r = torch.from_numpy(np.stack(batch.reward)).float()
        mask = torch.Tensor(np.stack(batch.mask))

        v = self.critic(s).squeeze(-1).detach()
        log_pi_old_sa = self.actor.get_log_prob(s, a).detach()
        A_sa, v_target = self.estimate_advantage(r, v, mask)

        for i in range(10):
            perm = torch.randperm(batch_size)
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = v_target[perm], A_sa[perm], s[perm], a[perm], \
                                                                           log_pi_old_sa[perm]
            optim_chunk_num = int(math.ceil(batch_size * 1.0 / 64))
            v_target_shuf, A_sa_shuf, s_shuf, a_shuf, log_pi_old_sa_shuf = torch.chunk(v_target_shuf, optim_chunk_num), \
                                                                           torch.chunk(A_sa_shuf, optim_chunk_num), \
                                                                           torch.chunk(s_shuf, optim_chunk_num), \
                                                                           torch.chunk(a_shuf, optim_chunk_num), \
                                                                           torch.chunk(log_pi_old_sa_shuf,
                                                                                       optim_chunk_num)
            actor_loss, critic_loss = 0., 0.
            for v_target_b, A_sa_b, s_b, a_b, log_pi_old_sa_b in zip(v_target_shuf, A_sa_shuf, s_shuf, a_shuf,
                                                                     log_pi_old_sa_shuf):
                # 1. update critic network
                self.critic_optim.zero_grad()
                v_b = self.critic(s_b).squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                critic_loss += loss.item()

                # backprop
                loss.backward()
                # nn.utils.clip_grad_norm(self.value.parameters(), 4)
                self.critic_optim.step()

                # 2. update actor network by clipping
                self.actor_optim.zero_grad()
                # [b, 1]
                log_pi_sa = self.actor.get_log_prob(s_b, a_b)
                # ratio = exp(log_Pi(a|s) - log_Pi_old(a|s)) = Pi(a|s) / Pi_old(a|s)
                # we use log_pi for stability of numerical operation
                # [b, 1] => [b]
                ratio = (log_pi_sa - log_pi_old_sa_b).exp().squeeze(-1)
                # because the joint action prob is the multiplication of the prob of each da
                # it may become extremely small
                # and the ratio may be inf in this case, which causes the gradient to be nan
                # clamp in case of the inf ratio, which causes the gradient to be nan
                ratio = torch.clamp(ratio, 0, 10)
                surrogate1 = ratio * A_sa_b
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A_sa_b
                # this is element-wise comparing.
                # we add negative symbol to convert gradient ascent to gradient descent
                surrogate = - torch.min(surrogate1, surrogate2).mean()
                actor_loss += surrogate.item()

                # backprop
                surrogate.backward()
                # although the ratio is clamped, the grad may still contain nan due to 0 * inf
                # set the inf in the gradient to 0
                for p in self.actor.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                # self.lock.acquire() # retain lock to update weights
                self.actor_optim.step()
                # self.lock.release() # release lock

            critic_loss /= optim_chunk_num
            actor_loss /= optim_chunk_num
            accurate = self.calculate_accurate(s, a)
            logging.debug('training PPO: iteration {}'.format(i))
            logging.debug(
                '              critic_loss {:.05f}, actor_loss {:.05f}, same actions {:.05f}'.format(critic_loss,
                                                                                                     actor_loss,
                                                                                                     accurate))

    def imitate(self):
        batch = self.memory.get_batch()
        batch_size = len(self.memory)
        self.memory.clear()
        s = torch.from_numpy(np.stack(batch.state)).float()
        a = torch.from_numpy(np.stack(batch.action))

        # train actor
        logging.debug('PPO: imitate to {} samples'.format(batch_size))
        for i in range(10):
            perm = torch.randperm(batch_size)
            s_shuf, a_shuf = s[perm], a[perm]
            optim_chunk_num = int(math.ceil(batch_size * 1.0 / 64))
            s_shuf, a_shuf = torch.chunk(s_shuf, optim_chunk_num), torch.chunk(a_shuf, optim_chunk_num)
            actor_loss = 0
            for s_b, a_b in zip(s_shuf, a_shuf):
                self.actor_optim.zero_grad()
                a_b_pred = self.actor(s_b)
                loss = F.cross_entropy(a_b_pred, a_b)
                actor_loss += loss
                loss.backward()
                self.actor_optim.step()
            accurate = self.calculate_accurate(s, a)
            logging.debug('PPO: accurate {:.3f}, imitate loss {:.05f}'.format(accurate, actor_loss))

    def calculate_accurate(self, state, action):
        a_pred_weigh = self.actor(state)
        a_pred = torch.argmax(a_pred_weigh, dim=-1)
        accurate = 1.0 * torch.sum(a_pred == action).item() / len(action)
        return accurate

    def predict(self, state):
        return self.actor.select_action(state)

    def estimate_advantage(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz)
        delta = torch.Tensor(batchsz)
        A_sa = torch.Tensor(batchsz)

        prev_v_target = 0.
        prev_v = 0.
        prev_A_sa = 0.
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]

        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + '.critic')
        torch.save(self.actor.state_dict(), filename + '.actor')

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + '.critic'))
        self.actor.load_state_dict(torch.load(filename + '.actor'))


class DiscreteActor(nn.Module):
    def __init__(self, s_dim, a_dim, h_dim=64):
        super(DiscreteActor, self).__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, h_dim),
                                 nn.ReLU(),
                                 nn.Linear(h_dim, a_dim))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        s = torch.tensor(s).float()
        a_weights = self.net(s)
        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: int
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, dim=-1)
        if sample:
            a = torch.multinomial(a_probs, 1, replacement=False)
        else:
            a = torch.argmax(a_probs)
        assert a.shape == (1, 1)
        action = a.squeeze().item()
        return action

    def get_log_prob(self, s, a):  # todo
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.softmax(a_weights, dim=-1)
        a_probs = a_probs.gather(-1, a.unsqueeze(-1)).squeeze()
        return torch.log(a_probs)


class Critic(nn.Module):
    def __init__(self, s_dim, hv_dim=64):
        super(Critic, self).__init__()

        self.net = nn.Sequential(nn.Linear(s_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, hv_dim),
                                 nn.ReLU(),
                                 nn.Linear(hv_dim, 1))

    def forward(self, s):
        """
        :param s: [b, s_dim]
        :return:  [b, 1]
        """
        value = self.net(s)

        return value


PPOTransition = namedtuple('Transition', ('state', 'action', 'reward', 'mask'))


class PPOMemory(object):
    def __init__(self):

        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(PPOTransition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return PPOTransition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return PPOTransition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def clear(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
