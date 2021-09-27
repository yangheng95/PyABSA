# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os
import pickle
import time

import requests
import torch
from autocuda import auto_cuda, auto_cuda_name
from findfile import find_files
from termcolor import colored
from functools import wraps

from update_checker import parse_version

from pyabsa import __version__

SENTIMENT_PADDING = -999


def save_args(config, save_path):
    f = open(os.path.join(save_path), mode='w', encoding='utf8')
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write('{}: {}\n'.format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None, mode=0):
    activated_args = []
    default_args = []
    for arg in config.args:
        if config.args_call_count[arg]:
            activated_args.append('>>> {0}: {1}  --> Active'.format(arg, config.args[arg]))
        else:
            if mode == 0:
                default_args.append('>>> {0}: {1}  --> Default'.format(arg, config.args[arg]))
            else:
                default_args.append('>>> {0}: {1}  --> Not Used'.format(arg, config.args[arg]))

    for line in activated_args:
        if logger:
            logger.info(line)
        else:
            print(colored(line, 'green'))

    for line in default_args:
        if logger:
            logger.info(line)
        else:
            print(colored(line, 'yellow'))


def check_and_fix_labels(label_set, label_name, all_data, opt):
    # update polarities_dim, init model behind this function!
    p_min, p_max = min(label_set), max(label_set)
    if not sorted(list(label_set)) == sorted(list(range(p_max - p_min + 1))) or \
            len(sorted(list(label_set))) != len(sorted(list(range(p_max - p_min + 1)))):
        print('Warning! Invalid label detected, label-fixing is triggered! (You can manually refactor the labels instead.)')
        new_label_dict = {origin_label: idx for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        origin_label_map = {idx: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        if 'origin_label_map' not in opt.args:
            opt.origin_label_map = origin_label_map

        if opt.origin_label_map != origin_label_map:
            raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')

        for item in all_data:
            try:
                item[label_name] = new_label_dict[item[label_name]]
            except:
                item.polarity = new_label_dict[item.polarity]
        print('original labels:{}'.format(list(label_set)))
        print('mapped new labels:{}'.format(new_label_dict))
    else:
        opt.origin_label_map = {idx: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}


def get_device(auto_device):
    if isinstance(auto_device, str):
        device = auto_device
    elif isinstance(auto_device, bool):
        device = auto_cuda() if auto_device else 'cpu'
    else:
        device = auto_cuda()
        try:
            torch.device(device)
        except RuntimeError as e:
            print('Device assignment error: {}, redirect to CPU'.format(e))
            device = 'cpu'
    device_name = auto_cuda_name()
    return device, device_name


def load_checkpoint(trainer, from_checkpoint_path):
    if from_checkpoint_path:
        model_path = find_files(from_checkpoint_path, '.model')
        state_dict_path = find_files(from_checkpoint_path, '.state_dict')
        config_path = find_files(from_checkpoint_path, '.config')

        if from_checkpoint_path:
            if not config_path:
                raise FileNotFoundError('.config file is missing!')
            config = pickle.load(open(config_path[0], 'rb'))
            if model_path:
                if config.model != trainer.opt.model:
                    print(colored('Warning, the checkpoint was not trained using {} from param_dict'.format(trainer.opt.model.__name__)), 'red')
                trainer.model = torch.load(model_path[0])
            if state_dict_path:
                trainer.model.load_state_dict(torch.load(state_dict_path[0]))
                trainer.model.opt = trainer.opt
                trainer.model.to(trainer.opt.device)
            else:
                print('.model or .state_dict file is missing!')
        else:
            print('No checkpoint found in {}'.format(from_checkpoint_path))
        print('Checkpoint loaded!')


def retry(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print('Catch exception: {} in {}, retry soon if you dont terminate process...'.format(e, f))
                time.sleep(5)

    return decorated


def save_json(dic, save_path):
    if isinstance(dic, str):
        dic = eval(dic)
    with open(save_path, 'w', encoding='utf-8') as f:
        # f.write(str(dict))
        str_ = json.dumps(dic, ensure_ascii=False)
        f.write(str_)


def load_json(save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = f.readline().strip()
        print(type(data), data)
        dic = json.loads(data)
    return dic


def validate_version():
    try:
        response = requests.get("https://pypi.org/pypi/pyabsa/json", timeout=1)
    except requests.exceptions.RequestException:
        return
    if response.status_code == 200:
        data = response.json()
        versions = list(data["releases"].keys())
        versions.sort(key=parse_version, reverse=True)
        if __version__ not in versions:
            print(colored('You are using a DEPRECATED / TEST version of PyABSA which may contain severe bug!'
                          ' Please update using pip install -U pyabsa!', 'red'))


optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW
}
