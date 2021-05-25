# encoding:utf-8

import logging
import time
import os


def init_logging(log_dir='./log', filename='output.log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, filename), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(message)s\n                       - %(levelname)s %(asctime)s %(filename)s(%(funcName)s %(lineno)d)'))

    # stream handle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # setting logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)


init_logging(filename='run.log')
