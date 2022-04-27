# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os
import pickle
import threading
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
    args = [key for key in sorted(config.args.keys())]
    for arg in args:
        if logger:
            logger.info('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))
        else:
            print('{0}:{1}\t-->\tCalling Count:{2}'.format(arg, config.args[arg], config.args_call_count[arg]))


def validate_example(text: str, aspect: str, polarity: str):
    if len(text) < len(aspect):
        raise ValueError(colored('AspectLengthExceedTextError -> <aspect: {}> is longer than <text: {}>, <polarity: {}>'.format(aspect, text, polarity), 'red'))

    if aspect.strip().lower() not in text.strip().lower():
        raise ValueError(colored('AspectNotInTextError -> <aspect: {}> is not in <text: {}>>'.format(aspect, text), 'yellow'))

    warning = False

    if len(aspect.split(' ')) > 10:
        print(colored('AspectTooLongWarning -> <aspect: {}> is too long, <text: {}>, <polarity: {}>'.format(aspect, text, polarity), 'yellow'))
        warning = True

    if len(polarity.split(' ')) > 3:
        print(colored('LabelTooLongWarning -> <label: {}> is too long, <text: {}>, <aspect: {}>'.format(polarity, text, aspect), 'yellow'))
        warning = True

    return warning


def check_and_fix_labels(label_set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    index_to_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_label' not in opt.args:
        opt.index_to_label = index_to_label
        opt.label_to_index = label_to_index

    if opt.index_to_label != index_to_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_label.update(index_to_label)
        opt.label_to_index.update(label_to_index)

    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))


def check_and_fix_IOB_labels(label_map, opt):
    # update polarities_dim, init model behind execution of this function!
    index_to_IOB_label = {int(label_map[origin_label]): origin_label for origin_label in label_map}
    opt.index_to_IOB_label = index_to_IOB_label


def get_device(auto_device):
    if isinstance(auto_device, str) and auto_device == 'allcuda':
        device = 'cuda'
    elif isinstance(auto_device, str):
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


def resume_from_checkpoint(trainer, from_checkpoint_path):
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
                if torch.cuda.device_count() > 1 and trainer.opt.device == 'allcuda':
                    trainer.model.module.load_state_dict(torch.load(state_dict_path[0]))
                else:
                    trainer.model.load_state_dict(torch.load(state_dict_path[0]))
                trainer.model.opt = trainer.opt
                trainer.model.to(trainer.opt.device)
            else:
                print('.model or .state_dict file is missing!')
        else:
            print('No checkpoint found in {}'.format(from_checkpoint_path))
        print('Resume training from Checkpoint: {}!'.format(from_checkpoint_path))


class TransformerConnectionError(ValueError):
    def __init__(self):
        pass


def retry(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        count = 5
        while count:

            try:
                return f(*args, **kwargs)
            except (
                TransformerConnectionError,
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ProxyError,
                requests.exceptions.SSLError,
                requests.exceptions.BaseHTTPError,
            ) as e:
                print('Training Exception: {}, will retry later'.format(e))
                time.sleep(60)
                count -= 1

    return decorated


def time_out(interval=5, callback=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=func, args=args, kwargs=kwargs)
            t.setDaemon(True)  # 设置主线程结束子线程立刻结束
            t.start()
            t.join(interval)  # 主线程阻塞等待interval秒
            if t.is_alive() and callback:
                return threading.Timer(0, callback).start()  # 立即执行回调函数
            else:
                return

        return wrapper

    return decorator


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


def validate_pyabsa_version():
    try:
        response = requests.get("https://pypi.org/pypi/pyabsa/json", timeout=1)
    except requests.exceptions.RequestException:
        return
    if response.status_code == 200:
        data = response.json()
        versions = list(data["releases"].keys())
        versions.sort(key=parse_version, reverse=True)
        if __version__ not in versions:
            print(colored('You are using a DEPRECATED or TEST version of PyABSA. Consider update using pip install -U pyabsa!', 'red'))


optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,
    # 'radam': torch.optim.Adam if torch.version.__version__ <= '1.9.1' else torch.optim.RAdam,
    # 'nadam': torch.optim.Adam if torch.version.__version__ <= '1.9.1' else torch.optim.NAdam,
    # 'sparseadam': torch.optim.Adam if torch.version.__version__ <= '1.9.1' else torch.optim.SparseAdam,
    torch.optim.Adadelta: torch.optim.Adadelta,  # default lr=1.0
    torch.optim.Adagrad: torch.optim.Adagrad,  # default lr=0.01
    torch.optim.Adam: torch.optim.Adam,  # default lr=0.001
    torch.optim.Adamax: torch.optim.Adamax,  # default lr=0.002
    torch.optim.ASGD: torch.optim.ASGD,  # default lr=0.01
    torch.optim.RMSprop: torch.optim.RMSprop,  # default lr=0.01
    torch.optim.SGD: torch.optim.SGD,
    torch.optim.AdamW: torch.optim.AdamW,
    # torch.optim.RAdam: torch.optim.RAdam,
    # torch.optim.NAdam: torch.optim.NAdam,
    # torch.optim.SparseAdam: torch.optim.SparseAdam,
}
