# encoding:utf-8
import re
from matplotlib import pyplot

origin_filename = '/home/huangchenping/ddq_src/backup/baseline_ddq_k5_5_agent_800_epoches/run1' \
                  '/pool_trend.txt'
reduce_file = '/home/huangchenping/ddq_src/backup/reduce_world_pool/run1/pool_trend.txt'

with open(origin_filename, 'r') as f:
    content = f.read()
pattern = 'episode (\d+)\D*(\d+)\D*(\d+)'
result = re.findall(pattern, content)[:400]
episode_list = [values[0] for values in result]
user_pool_list = [values[1] for values in result]
world_pool_list = [values[2] for values in result]

with open(reduce_file, 'r') as f:
    content = f.read()
pattern = 'episode (\d+)\D*(\d+)\D*(\d+)'
result = re.findall(pattern, content)
reduce_world_pool_list = [values[2] for values in result]

# 画图
pyplot.plot(episode_list, user_pool_list, label='replay pool (user)', color='#ED7D31')
pyplot.plot(episode_list, world_pool_list, label='replay pool (world model)', color='#BF9000')

pyplot.plot(episode_list, reduce_world_pool_list, label='change max size: replay pool (world '
                                                        'model)', color='green')

pyplot.xlabel('episode')
pyplot.ylabel('pool size')
pyplot.legend()
pyplot.show()
