# -*- coding: utf-8 -*-
# file: functional.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import copy
from argparse import Namespace

from pyabsa.utils.pyabsa_utils import get_auto_device, detect_dataset

from pyabsa.apc.inferring.sentiment_classifier import SentimentClassifier
from pyabsa.apc.training.apc_trainer import train4apc

from pyabsa.atepc.training.atepc_trainer import train4atepc
from pyabsa.atepc.inferring.aspect_extractor import AspectExtractor

from pyabsa.config.atepc_config import atepc_param_dict_base
from pyabsa.config.apc_config import apc_param_dict_base


choice = get_auto_device()


def init_config(config_dict, base_config_dict, auto_device=True):
    if config_dict:
        if auto_device and 'device' not in config_dict:
            if choice >= 0:
                base_config_dict['device'] = 'cuda:' + str(choice)
            else:
                base_config_dict['device'] = 'cpu'
        if not auto_device and 'device' not in config_dict:
            base_config_dict['device'] = 'cpu'
        # reload hyper-parameter from parameter dict
        for key in config_dict:
            base_config_dict[key] = config_dict[key]

    apc_config = Namespace(**base_config_dict)

    if 'lcfs' in apc_config.model_name or apc_config.use_syntax_based_SRD:
        print('-' * 130)
        print('(Force to) use syntax distance-based semantic-relative distance,'
              ' however Chinese is not supported to parse syntax distance yet!')
        print('-' * 130)

    return apc_config


def train_apc(parameter_dict=None,
              dataset_path=None,
              model_path_to_save=None,
              auto_evaluate=True,
              auto_device=True):
    '''
    evaluate model performance while training_tutorials model in order to obtain best benchmarked model
    '''
    # load training_tutorials set

    dataset_file = detect_dataset(dataset_path, auto_evaluate, task='apc')

    config = init_config(parameter_dict, apc_param_dict_base, auto_device)
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
                              sentiment_map=None,
                              auto_device=True):
    if trained_model_path and os.path.isdir(trained_model_path):
        infer_model = SentimentClassifier(trained_model_path, sentiment_map=sentiment_map)
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
    evaluate model performance while training_tutorials model in order to obtain best benchmarked model
    '''
    # load training_tutorials set

    dataset_file = detect_dataset(dataset_path, auto_evaluate, task='atepc')

    config = init_config(parameter_dict, atepc_param_dict_base, auto_device)
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
                          sentiment_map=None,
                          auto_device=True):
    if trained_model_path and os.path.isdir(trained_model_path):
        aspect_extractor = AspectExtractor(trained_model_path, sentiment_map=sentiment_map)
        aspect_extractor.to('cuda:' + str(choice)) if auto_device and choice >= 0 else aspect_extractor.cpu()
        return aspect_extractor
    else:
        raise RuntimeError('Not a valid model path!')
