# encoding:utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cPickle as pickle
from torch.utils.data import TensorDataset, DataLoader

from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import logging


class ImitationNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(ImitationNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.net = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.output_size), )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        y = self.forward(x)
        return torch.argmax(y, 1)


class ImitationPolicy():
    def __init__(self, input_size=10, output_size=10, device='cpu'):
        # 加载数据
        # 设置优化器，损失
        self.device = torch.device(device)
        self.net = ImitationNet(input_size, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=1e-3)
        self.data_loader = None

    def create_data_loader(self, s_a_path=None):
        s_a_dataset = pickle.load(open(s_a_path, 'rb'))
        state_set = s_a_dataset['state'].float()
        action_set = s_a_dataset['action']
        data_set = TensorDataset(state_set, action_set)
        logging.info('{} state-action pairs'.format(len(data_set)))
        self.data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)

    def train(self, epochs=400):
        for epoch in xrange(epochs):
            # print 'epoch:', epoch
            total_loss = 0
            total_samples = 0
            correct = 0
            with tqdm(total=len(self.data_loader.dataset), ncols=120) as t:
                for i, data in enumerate(self.data_loader):
                    state_data, action_data = data[0].to(self.device), data[1].to(self.device)
                    self.optimizer.zero_grad()

                    output_data = self.net(state_data)
                    loss = self.criterion(output_data, action_data)
                    loss.backward()
                    self.optimizer.step()

                    total_samples += len(state_data)
                    correct += torch.sum(torch.argmax(output_data, dim=-1) == action_data).item()
                    total_loss += loss.item()

                    t.set_description('Epoch {}'.format(epoch))
                    t.set_postfix(acc=correct * 1.0 / total_samples,
                                  loss=total_loss * 1.0 / total_samples)
                    t.update(len(state_data))
                    if i + 1 == len(self.data_loader):
                        logging.info('im model train: epoch {}, accuracy {}, loss {}'.format(
                            epoch, correct * 1.0 / total_samples, total_loss * 1.0 / total_samples))

    # 训练模型，打印模型精度

    def state_to_action(self, state):
        pass

    def save(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

    def predict(self, x):
        # 直接得到对应的动作标签
        return self.net.predict(x.to(self.device))

    def forward(self, x):
        # 计算所有Q值
        return self.net.forward(x.to(self.device))

    def freeze(self, flag=True):
        for param in self.net.parameters():
            param.requires_grad = flag


# 返回动作, state: [1,273], state: [1]

# 用 user simulator 评估 IM 模型


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    im = ImitationPolicy(device='cuda')
    models = []
    for i in range(10):
        import copy

        print i
        models.append(copy.deepcopy(im))
