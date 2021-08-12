import numpy as np
import torch
import json
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random

# filename = 'result/dqn_800_k1_no_shaping/performance.json'
# old_data = json.load(open(filename, 'r'))
# net_data = OrderedDict()
# for key, d in old_data.items():
#     d = sorted(d.items(), key=lambda pair: int(pair[0]))
#     net_data[key] = OrderedDict(d)
# json.dump(net_data, open(filename, 'w'), indent=4)

data = [(i, torch.tensor([i]), np.array(10))for i in range(100)]
sample = random.sample(data,3)
print sample

