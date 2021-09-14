# encoding:utf-8
import os, logging, sys
from functools import wraps
import time


def init_logging(filepath='./log/output.log', log_level=logging.DEBUG):
    # file handler
    abs_path = os.path.abspath(filepath)
    dir_name = os.path.dirname(abs_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    formatter = logging.Formatter(
        "%(levelname)9s %(asctime)s: %(filename)-20s (%(funcName)-20s %(lineno)4d) |  %(message)s",
        datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(filepath, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # stream handle
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # setting logging
    logging.getLogger().setLevel(log_level)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().addHandler(console_handler)

    logging.debug('current work path: {}'.format(os.getcwd()))
    logging.debug('command: {} -u {}'.format(sys.executable, ' '.join(sys.argv)))
    logging.debug('write log to {}\n'.format(abs_path))


# 装饰器：在定义其他函数时在前一行加入 "@calculate_time"
def calculate_time(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        start = time.time()
        _result = func(*args, **kwargs)
        total_time = time.time() - start

        rest = total_time - int(total_time)
        total_time = int(total_time)
        logging.debug(
            'Running function \"{}\" spent time {:02}:{:02}:{:02}.{:03}'.format(func.__name__,
                                                                                total_time // 3600,
                                                                                total_time % 3600 // 60,
                                                                                total_time % 60,
                                                                                int(rest * 1000)))
        return _result

    return wrapped_function
