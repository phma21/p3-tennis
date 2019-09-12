#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import uuid
from copy import deepcopy

import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
from pathlib import Path
from deep_rl.component.envs import _reacher_instance


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save(norm_and_join(config.output_dir, 'data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps)))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            if config.game == 'reacher':
                agent.task.env.envs[0].env.train_mode = False
            agent.eval_episodes()
            if config.game == 'reacher':
                agent.task.env.envs[0].env.train_mode = True
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()


def norm_and_join(basepath, subpath):
    return os.path.normpath(os.path.join(basepath, subpath))


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def generate_tag(params):
    if 'tag' in params.keys():
        return
    params_copy = deepcopy(params)
    del params_copy['game']
    del params_copy['output_dir']

    game = params['game']
    string = ['%s_%s' % (k, v) for k, v in sorted(params_copy.items())]
    tag = '%s--%s-%s' % (game, datetime.datetime.now().strftime("%y%m%d-%H%M%S"), '-'.join(string))
    params['tag'] = tag


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
