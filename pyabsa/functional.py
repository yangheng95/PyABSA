# -*- coding: utf-8 -*-
# file: functional.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import copy
from argparse import Namespace

from pyabsa.apc.prediction.prediction import SentimentClassifier
from pyabsa.apc.training.apc_trainer import apc_trainer
from pyabsa.apc.training.training_configs import *
from pyabsa.pyabsa_utils import find_target_file, get_auto_device
from pyabsa.apc.dataset_utils.apc_utils import parse_experiments

choice = get_auto_device()


def init_training_config(config_dict, auto_device=True):
    config = dict()
    config['SRD'] = SRD
    config['batch_size'] = batch_size
    config['eta'] = eta
    config['dropout'] = dropout
    config['l2reg'] = l2reg
    config['lcf'] = lcf
    config['initializer'] = initializer
    config['learning_rate'] = learning_rate
    config['max_seq_len'] = max_seq_len
    config['model_name'] = model_name
    config['num_epoch'] = num_epoch
    config['optimizer'] = optimizer
    config['pretrained_bert_name'] = pretrained_bert_name
    config['use_bert_spc'] = use_bert_spc
    config['use_dual_bert'] = use_dual_bert
    config['window'] = window
    config['seed'] = seed
    config['embed_dim'] = embed_dim
    config['hidden_dim'] = hidden_dim
    config['polarities_dim'] = polarities_dim
    config['sigma'] = sigma
    config['log_step'] = log_step

    # reload hyper-parameter from training config
    path = os.path.abspath(__file__)
    folder = os.path.dirname(path)
    config_path = os.path.join(folder, 'apc/training/training_configs.json')
    _config = vars(parse_experiments(config_path)[0])
    for key in config:
        _config[key] = config[key]

    if not config_dict:
        config_dict = dict()
    # reload hyper-parameter from parameter dict
    for key in config_dict:
        _config[key] = config_dict[key]

    if auto_device and 'device' not in _config:
        if choice >= 0:
            _config['device'] = 'cuda:' + str(choice)
        else:
            _config['device'] = 'cpu'

    _config = Namespace(**_config)

    return _config


def train_apc(parameter_dict=None,
              dataset_path=None,
              model_path_to_save=None,
              auto_evaluate=True,
              auto_device=True):
    '''
    evaluate model performance while training model in order to obtain best benchmarked model
    '''
    # load training set
    try:
        dataset_file = dict()
        dataset_file['train'] = find_target_file(dataset_path, 'train', exclude_key='infer')
        if auto_evaluate and find_target_file(dataset_path, 'test', exclude_key='infer'):
            dataset_file['test'] = find_target_file(dataset_path, 'test', exclude_key='infer')
        if auto_evaluate and not find_target_file(dataset_path, 'test', exclude_key='infer'):
            print('Cna not find test set using for evaluating!')
    except:
        raise RuntimeError('Can not load train set or test set! '
                           'Make sure there are (only) one train set and one test set in the path:', dataset_path)

    config = init_training_config(parameter_dict, auto_device)
    config.dataset_path = dataset_path
    config.model_path_to_save = model_path_to_save
    config.dataset_file = dataset_file
    model_path = []

    if not isinstance(config.seed, int) and 'test' in dataset_file:
        for _, s in enumerate(config.seed):
            t_config = copy.deepcopy(config)
            t_config.seed = s
            model_path.append(apc_trainer(t_config))
        return SentimentClassifier(from_model_path=max(model_path))
    elif 'test' in dataset_file:  # Avoid multiple training without evaluating
        return SentimentClassifier(from_model_path=apc_trainer(config))
    else:  # Avoid evaluating without test set
        return SentimentClassifier(from_training=apc_trainer(config))


def load_trained_model(trained_model_path=None,
                       auto_device=False):
    if trained_model_path and os.path.isdir(trained_model_path):
        infer_model = SentimentClassifier(trained_model_path)
        infer_model.to('cuda:' + str(choice)) if auto_device and choice >= 0 else infer_model.cpu()
        return infer_model
    else:
        raise RuntimeError('Not a valid model path!')
