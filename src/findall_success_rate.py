import os
import re
from collections import OrderedDict

top_dir = './result/dqn_p4_epoch400/'
log_file = 'validation.log'
all_rate = OrderedDict()
for i in range(1, 6):
    sub_dir = 'run' + str(i)
    filepath = os.path.join(top_dir, sub_dir, log_file)
    with open(filepath, 'r') as f:
        content = f.read()
        results = re.findall("(\S+) model.*?success_rate.*?(0\.\d+)", content)
        # print results
        for method, rate in results:
            if method not in all_rate.keys():
                all_rate[method] = []
            all_rate[method].append(float(rate))
# print all_rate
for method, rates in all_rate.items():
    # print rates
    # print [str(rate) for rate in rates]
    print '{} ({})/5={}'.format(method, '+'.join(['{:.3f}'.format(rate) for rate in rates]),
                                sum(rates) / 5)
