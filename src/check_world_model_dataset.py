import logging
import pandas as pd
import os
import torch
from deep_dialog.utils import calculate_time
from tqdm import tqdm

logging.getLogger().setLevel(logging.DEBUG)


@calculate_time
def create_cluster_(dataset):
    state, agent_action, reward, term, user_action = dataset
    rest = set(range(len(state)))
    cluster = dict((i, [i, ]) for i in range(len(state)))
    for i in range(len(state)):
        if i in rest:
            for j in range(i + 1, len(state)):
                if j in rest:
                    if agent_action[i].item() == agent_action[j].item() and torch.sum(
                            state[i] == state[j]).item() == 330:
                        rest.remove(j)
                        cluster[i].append(j)
                        del cluster[j]
            rest.remove(i)
    return cluster


def create_cluster(dataset):
    state, agent_action, reward, term, user_action = dataset
    cluster = dict((i, [i, ]) for i in range(len(state)))
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if agent_action[i].item() == agent_action[j].item() and torch.sum(
                    state[i] == state[j]).item() == state.shape[-1]:
                cluster[i].append(j)
                # print 'delect', j
                # cluster.pop(j)
    return cluster


def calculate_dataset_limit(cluster, dataset):
    state, agent_action, reward, term, user_action = dataset
    reward_conflict = 0
    term_conflict = 0
    user_action_conflict = 0
    for i, other in cluster.items():
        for j in other[1:]:
            if not reward[i].item() == reward[j].item():
                reward_conflict += 1
            if not term[i].item() == term[j].item():
                term_conflict += 1
            if not user_action[i].item() == user_action[j].item():
                user_action_conflict += 1

    reward_limit = 1 - reward_conflict * 1.0 / len(state)
    term_limit = 1 - term_conflict * 1.0 / len(state)
    user_action_limit = 1 - user_action_conflict * 1.0 / len(state)

    return reward_limit, term_limit, user_action_limit


@calculate_time
def calculate_limit(dir_name="result/_DPPO/wm_dataset", filename='wm_memory_{}.pt', epochs=100,
                    write_model_dir='./result/_DPPO/wm_dataset'):
    wm_train_res = pd.DataFrame(columns=['epoch', 'acc_reward', 'acc_term', 'acc_user_action'])
    for episode in tqdm(range(epochs)):
        print '\n\n'
        filepath = os.path.join(dir_name, filename.format(episode))
        logging.debug('Load dataset from {}'.format(os.path.abspath(filepath)))
        d = torch.load(filepath)
        clus = create_cluster(d)
        reward_limit, term_limit, user_action_limit = calculate_dataset_limit(clus, d)
        logging.debug('Episode {:03}: reward {:.3f}, term {:.3f}, u_action {:.3f}'.format(episode,
                                                                                          reward_limit,
                                                                                          term_limit,
                                                                                          user_action_limit))
        wm_train_res = wm_train_res.append(
            {'epoch': episode, 'acc_reward': reward_limit, 'acc_term': term_limit,
             'acc_user_action': user_action_limit}, ignore_index=True)

    dir_name = os.path.join(write_model_dir, 'acc_limit', 'run1')
    csv_path = os.path.join(dir_name, 'world_model_train_result.csv')
    wm_train_res['epoch'] = wm_train_res['epoch'].astype(int)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    wm_train_res.to_csv(csv_path, index=False)
    logging.debug('Save World Model train records to {}'.format(os.path.abspath(csv_path)))


calculate_limit(epochs=100)
