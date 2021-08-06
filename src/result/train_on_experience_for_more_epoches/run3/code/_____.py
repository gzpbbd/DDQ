# encoding:utf-8

import torch
from torch import nn
import copy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)


if __name__ == '__main__':

    # 获取设备
    device = torch.device('cuda:0')
    print(device)


    # 实例化网络，置于GPU上
    net = Net()
    net.to(device)
    print ("moving net to gpu done")


    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    best = copy.deepcopy(net)
    print ('deepcopy done')
