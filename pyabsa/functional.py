# -*- coding: utf-8 -*-
# file: functional.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import copy
from argparse import Namespace

from pyabsa.pyabsa_utils import find_target_file, get_auto_device

from pyabsa.apc.inferring.SentimentClassifier import SentimentClassifier
from pyabsa.apc.training.apc_trainer import train4apc
from pyabsa.apc.training import apc_config
from pyabsa.apc.dataset_utils.apc_utils import parse_apc_params

from pyabsa.atepc.training.atepc_trainer import train4atepc
from pyabsa.atepc.inferring.AspectExtractor import AspectExtractor
from pyabsa.atepc.training import atepc_config

choice = get_auto_device()


def init_apc_config(config_dict, auto_device=True):
    config = dict()
    config['model_name'] = apc_config.model_name
    config['SRD'] = apc_config.SRD
    config['batch_size'] = apc_config.batch_size
    config['eta'] = apc_config.eta
    config['dropout'] = apc_config.dropout
    config['l2reg'] = apc_config.l2reg
    config['lcf'] = apc_config.lcf
    config['initializer'] = apc_config.initializer
    config['learning_rate'] = apc_config.learning_rate
    config['max_seq_len'] = apc_config.max_seq_len
    config['num_epoch'] = apc_config.num_epoch
    config['optimizer'] = apc_config.optimizer
    config['pretrained_bert_name'] = apc_config.pretrained_bert_name
    config['use_bert_spc'] = apc_config.use_bert_spc
    config['use_dual_bert'] = apc_config.use_dual_bert
    config['window'] = apc_config.window
    config['seed'] = apc_config.seed
    config['embed_dim'] = apc_config.embed_dim
    config['hidden_dim'] = apc_config.hidden_dim
    config['polarities_dim'] = apc_config.polarities_dim
    config['sigma'] = apc_config.sigma
    config['log_step'] = apc_config.log_step
    config['loss_weight'] = apc_config.loss_weight

    # reload hyper-parameter from training config
    path = os.path.abspath(__file__)
    folder = os.path.dirname(path)
    config_path = os.path.join(folder, 'apc/training/training_configs.json')
    _config = vars(parse_apc_params(config_path)[0])
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


def init_atepc_config(config_dict, auto_device=True):
    config = dict()
    config['model_name'] = atepc_config.model_name
    config['SRD'] = atepc_config.SRD
    config['batch_size'] = atepc_config.batch_size
    config['dropout'] = atepc_config.dropout
    config['l2reg'] = atepc_config.l2reg
    config['lcf'] = atepc_config.lcf
    config['initializer'] = atepc_config.initializer
    config['learning_rate'] = atepc_config.learning_rate
    config['max_seq_len'] = atepc_config.max_seq_len
    config['num_epoch'] = atepc_config.num_epoch
    config['optimizer'] = atepc_config.optimizer
    config['pretrained_bert_name'] = atepc_config.pretrained_bert_name
    config['use_bert_spc'] = atepc_config.use_bert_spc
    config['use_dual_bert'] = atepc_config.use_dual_bert
    config['seed'] = atepc_config.seed
    config['embed_dim'] = atepc_config.embed_dim
    config['hidden_dim'] = atepc_config.hidden_dim
    config['polarities_dim'] = atepc_config.polarities_dim
    config['log_step'] = atepc_config.log_step
    config['gradient_accumulation_steps'] = atepc_config.gradient_accumulation_steps
    # # reload hyper-parameter from training config
    # path = os.path.abspath(__file__)
    # folder = os.path.dirname(path)
    # config_path = os.path.join(folder, 'atepc/training/experiments.json')
    # _config = vars(parse_apc_params(config_path)[0])
    # for key in config:
    #     _config[key] = config[key]

    if auto_device and 'device' not in config:
        if choice >= 0:
            config['device'] = 'cuda:' + str(choice)
        else:
            config['device'] = 'cpu'

    if not config_dict:
        config_dict = dict()
    # reload hyper-parameter from parameter dict
    for key in config_dict:
        config[key] = config_dict[key]

    _config = Namespace(**config)

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

    dataset_file = dict()
    dataset_file['train'] = find_target_file(dataset_path, 'train', exclude_key='infer')
    if auto_evaluate and find_target_file(dataset_path, 'test', exclude_key='infer'):
        dataset_file['test'] = find_target_file(dataset_path, 'test', exclude_key='infer')
    if auto_evaluate and not find_target_file(dataset_path, 'test', exclude_key='infer'):
        print('Cna not find test set using for evaluating!')
    if len(dataset_file) == 0:
        raise RuntimeError('Can not load train set or test set! '
                           'Make sure there are (only) one trainset and (only) one testset in the path:', dataset_path)

    config = init_apc_config(parameter_dict, auto_device)
    config.dataset_path = dataset_path
    config.model_path_to_save = model_path_to_save
    config.dataset_file = dataset_file
    model_path = []
    sent_classifier = None

    if isinstance(config.seed, int):
        config.seed = [config.seed]

    for _, s in enumerate(config.seed):
        t_config = copy.deepcopy(config)
        t_config.seed = s
        if model_path_to_save:
            model_path.append(train4apc(t_config))
        else:
            # always return the last trained model if dont save trained models
            sent_classifier = SentimentClassifier(model_arg=train4apc(t_config))
    if model_path_to_save:
        return SentimentClassifier(model_arg=max(model_path))
    else:
        return sent_classifier


def load_sentiment_classifier(trained_model_path=None,
                              auto_device=True):
    if trained_model_path and os.path.isdir(trained_model_path):
        infer_model = SentimentClassifier(trained_model_path)
        infer_model.to('cuda:' + str(choice)) if auto_device and choice >= 0 else infer_model.cpu()
        return infer_model
    else:
        raise RuntimeError('Not a valid model path!')


def train_atepc(parameter_dict=None,
                dataset_path=None,
                model_path_to_save=None,
                auto_evaluate=True,
                auto_device=True):
    '''
    evaluate model performance while training model in order to obtain best benchmarked model
    '''
    # load training set

    dataset_file = dict()
    dataset_file['train'] = find_target_file(dataset_path, 'train', exclude_key='infer')
    if auto_evaluate and find_target_file(dataset_path, 'test', exclude_key='infer'):
        dataset_file['test'] = find_target_file(dataset_path, 'test', exclude_key='infer')
    if auto_evaluate and not find_target_file(dataset_path, 'test', exclude_key='infer'):
        print('Cna not find test set using for evaluating!')
    if len(dataset_file) == 0:
        raise RuntimeError('Can not load train set or test set! '
                           'Make sure there are (only) one trainset and (only) one testset in the path:',
                           dataset_path)

    config = init_atepc_config(parameter_dict, auto_device)
    config.dataset_path = dataset_path
    config.model_path_to_save = model_path_to_save
    config.dataset_file = dataset_file
    model_path = []

    if isinstance(config.seed, int):
        config.seed = [config.seed]

    # always save all trained models in case of obtaining best performance in different metrics among ATE and APC task.
    for _, s in enumerate(config.seed):
        t_config = copy.deepcopy(config)
        t_config.seed = s
        model_path.append(train4atepc(t_config))
    
    return AspectExtractor(max(model_path))


def load_aspect_extractor(trained_model_path=None,
                          auto_device=True):
    if trained_model_path and os.path.isdir(trained_model_path):
        aspect_extractor = AspectExtractor(trained_model_path)
        aspect_extractor.to('cuda:' + str(choice)) if auto_device and choice >= 0 else aspect_extractor.cpu()
        return aspect_extractor
    else:
        raise RuntimeError('Not a valid model path!')
