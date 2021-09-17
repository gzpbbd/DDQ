import os.path

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
import pickle


class WorldModeSingleNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNet, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNet')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)

        reward = self.linear_h2r(h)
        term = torch.sigmoid(self.linear_h2t(h))
        action_weight = F.log_softmax(self.linear_h2a(h), -1)

        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetMoreFC(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetMoreFC, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        self.linear_h2h = nn.Linear(hidden_size, hidden_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetMoreFC')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)
        h = torch.tanh(self.linear_h2h(h))

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)
        h = torch.tanh(self.linear_h2h(h))

        reward = self.linear_h2r(h)
        term = torch.sigmoid(self.linear_h2t(h))
        action_weight = F.log_softmax(self.linear_h2a(h), -1)

        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetRelu(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetRelu, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetRelu')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.relu(h_s + h_a)

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.relu(h_s + h_a)

        reward = self.linear_h2r(h)
        term = torch.sigmoid(self.linear_h2t(h))
        action_weight = F.log_softmax(self.linear_h2a(h), -1)

        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetAddAttentionLayer(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetAddAttentionLayer, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        assert int(hidden_size ** 0.5) ** 2 == hidden_size
        self.attention_dim = int(hidden_size ** 0.5)
        self.attention_h2h = DotProductAttentionLayer(self.attention_dim, self.attention_dim)

        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetAddAttentionLayer')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)

        h_ = h.view(h.shape[0], self.attention_dim, self.attention_dim)
        h = self.attention_h2h(h_).view(h.shape[0], h.shape[1])

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward, term, action_weight = self.forward(s, a)
        term = torch.sigmoid(term)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetAddAttentionLayerDropout(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetAddAttentionLayerDropout, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        assert int(hidden_size ** 0.5) ** 2 == hidden_size
        self.attention_dim = int(hidden_size ** 0.5)
        self.attention_h2h = DotProductAttentionLayer(self.attention_dim, self.attention_dim)

        self.dropout = nn.Dropout(0.1)

        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)
        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetAddAttentionLayerDropout')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)

        h_ = h.view(h.shape[0], self.attention_dim, self.attention_dim)
        h = self.attention_h2h(self.dropout(h_)).view(h.shape[0], h.shape[1])
        h = self.dropout(h)

        reward = self.linear_h2r(h)
        term = self.linear_h2t(h)
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward, term, action_weight = self.forward(s, a)
        term = torch.sigmoid(term)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetAttentionReward(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetAttentionReward, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        assert int(hidden_size ** 0.5) ** 2 == hidden_size
        self.attention_dim = int(hidden_size ** 0.5)
        self.attention_r = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        # self.attention_a = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)

        self.linear_h2t = nn.Linear(hidden_size, termination_size)

        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetAttention Sigmoid')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)
        # h_r = torch.tanh(torch.cat([h_s.unsqueeze(1), h_a.unsqueeze(1)], dim=1))
        h_ = h.view(h.shape[0], self.attention_dim, self.attention_dim)

        attention_r_output = self.attention_r(h_).view(h.shape[0], h.shape[1])
        reward = self.linear_h2r(attention_r_output)
        term = self.linear_h2t(h)

        # attention_a_output = self.attention_a(h_).view(h.shape[0], h.shape[1])
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward, term, action_weight = self.forward(s, a)
        term = torch.sigmoid(term)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetAttentionRewardDropout(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetAttentionRewardDropout, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        self.dropout = nn.Dropout(0.1)

        assert int(hidden_size ** 0.5) ** 2 == hidden_size
        self.attention_dim = int(hidden_size ** 0.5)
        self.attention_r = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        # self.attention_a = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)

        self.linear_h2t = nn.Linear(hidden_size, termination_size)

        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetAttentionRewardDropout')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze()
        h = torch.tanh(h_s + h_a)
        # h_r = torch.tanh(torch.cat([h_s.unsqueeze(1), h_a.unsqueeze(1)], dim=1))
        h_ = h.view(h.shape[0], self.attention_dim, self.attention_dim)

        attention_r_output = self.attention_r(self.dropout(h_)).view(h.shape[0], h.shape[1])
        reward = self.linear_h2r(attention_r_output)
        term = self.linear_h2t(h)

        # attention_a_output = self.attention_a(h_).view(h.shape[0], h.shape[1])
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward, term, action_weight = self.forward(s, a)
        term = torch.sigmoid(term)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class WorldModeSingleNetAttentionRewardCat(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=64):
        super(WorldModeSingleNetAttentionRewardCat, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)
        self.agent_emb = nn.Embedding(agent_action_size, hidden_size)

        assert int(hidden_size ** 0.5) ** 2 == hidden_size
        self.attention_dim = int(hidden_size ** 0.5)
        self.attention_r = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        # self.attention_a = DotProductAttentionLayer(self.attention_dim, self.attention_dim)
        self.linear_h2r = nn.Linear(hidden_size * 2, reward_size)

        self.linear_h2t = nn.Linear(hidden_size, termination_size)

        self.linear_h2a = nn.Linear(hidden_size, user_action_size)
        logging.debug('World Model net: WorldModeSingleNetAttentionRewardCat')

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        h_s = self.linear_i2h(s)
        h_a = self.agent_emb(a).squeeze(1)
        h = torch.tanh(h_s + h_a)
        # reward
        h_r = torch.tanh(torch.cat([h_s.unsqueeze(1), h_a.unsqueeze(1)], dim=1))  # [n,2,h]
        h_ = h_r.view(h_r.shape[0], self.attention_dim * 2,
                      self.attention_dim)  # [n,2*h**0.5, h**0.5]
        # [n,2h]
        attention_r_output = self.attention_r(h_).view(h_r.shape[0], h_r.shape[1] * h_r.shape[2])
        reward = self.linear_h2r(attention_r_output)
        # term
        term = self.linear_h2t(h)
        # action
        # attention_a_output = self.attention_a(h_).view(h.shape[0], h.shape[1])
        action = F.log_softmax(self.linear_h2a(h), -1)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward, term, action_weight = self.forward(s, a)
        term = torch.sigmoid(term)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return reward, term, action


class DotProductAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DotProductAttentionLayer, self).__init__()
        self.wq = nn.Linear(in_features, out_features)
        self.wk = nn.Linear(in_features, out_features)
        self.wv = nn.Linear(in_features, out_features)
        self.dk = out_features

    def forward(self, input):
        q = self.wq(input)
        k = self.wk(input)
        v = self.wv(input)
        q_w = torch.matmul(q, k.transpose(-1, -2)) / (self.dk ** 0.5)
        output = torch.matmul(torch.softmax(q_w, dim=-1), v)
        return output


class WorldModelMultiNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 reward_size=1,
                 termination_size=1,
                 hidden_size=60):
        super(WorldModelMultiNet, self).__init__()

        self.reward_net = RewardNet(agent_action_size, state_size, reward_size, hidden_size)
        self.term_net = TermNet(agent_action_size, state_size, termination_size, hidden_size)
        self.action_net = ActionNet(agent_action_size, state_size, user_action_size, hidden_size)

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, user_action_size], float
        """
        reward = self.reward_net(s, a)
        term = self.reward_net(s, a)
        action = self.action_net(s, a)
        return reward, term, action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
            term: [n, 1], float
            action: [n, 1], int
        """
        reward = self.reward_net.predict(s, a)
        term = self.reward_net.predict(s, a)
        action = self.action_net.predict(s, a)
        return reward, term, action


class RewardNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 reward_size=1,
                 hidden_size=60):
        super(RewardNet, self).__init__()
        self.fc_s2h = nn.Linear(state_size, hidden_size)
        self.emb_u_a = nn.Embedding(agent_action_size, hidden_size)
        self.fc_h2o = nn.Linear(hidden_size, reward_size)

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        reward = self.fc_h2o(h)
        return reward

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            reward: [n, 1], float
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        reward = self.fc_h2o(h)
        return reward


class TermNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 termination_size=1,
                 hidden_size=60):
        super(TermNet, self).__init__()
        self.fc_s2h = nn.Linear(state_size, hidden_size)
        self.emb_u_a = nn.Embedding(agent_action_size, hidden_size)
        self.fc_h2o = nn.Linear(hidden_size, termination_size)

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            term: [n, 1], float
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        term = self.fc_h2o(h)
        return term

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            term: [n, 1], float
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        term = self.fc_h2o(h)
        return term


class ActionNet(nn.Module):
    def __init__(self,
                 agent_action_size,
                 state_size,
                 user_action_size,
                 hidden_size=60):
        super(ActionNet, self).__init__()
        self.fc_s2h = nn.Linear(state_size, hidden_size)
        self.emb_u_a = nn.Embedding(agent_action_size, hidden_size)
        self.fc_h2o = nn.Linear(hidden_size, user_action_size)

    def forward(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            action: [n, user_action_size], float
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        action = self.fc_h2o(h)
        return action

    def predict(self, s, a):
        """

        :param s: [n, state_size], float
        :param a: [n, 1], int
        :return:
            action: [n, 1], int
        """
        h_s = self.fc_s2h(s)
        h_a = self.emb_u_a(a).squeeze()
        h = torch.tanh(h_s + h_a)
        action_weight = F.log_softmax(self.fc_h2o(h), -1)
        action = action_weight.argmax(-1).unsqueeze(-1)
        return action
