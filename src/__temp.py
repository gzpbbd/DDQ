import sys
import os
from collections import OrderedDict
import re

dir_path = 'result/dqn_p4_epoch800/run1/'
with open(os.path.join(dir_path, 'train_rl.log'), 'r') as f:
    content = f.read()

for x in re.findall("Episode: (\d+)\n.*?\n.*?Simulation success rate ([0-9.]+)", content):
    print x
