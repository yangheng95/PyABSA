# -*- coding: utf-8 -*-
# file: functional.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

from argparse import Namespace

from pyabsa.dataset_utils import detect_dataset

from pyabsa.model_utils import APCModelList

from pyabsa.tasks.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.tasks.apc.training.apc_trainer import train4apc
from pyabsa.tasks.apc.baseline.glove_apc.training.apc_trainer_glove import train4apc_glove

from pyabsa.tasks.atepc.training.atepc_trainer import train4atepc
from pyabsa.tasks.atepc.prediction.aspect_extractor import AspectExtractor

from pyabsa.config.atepc_config import atepc_config_handler
from pyabsa.config.apc_config import apc_config_handler

from pyabsa.utils.logger import get_logger
from pyabsa.utils.pyabsa_utils import get_auto_device

from pyabsa import __version__

gpu_name, choice = get_auto_device()


def init_config(config_dict, base_config_dict, auto_device=True):
    if config_dict:
        if auto_device and 'device' not in config_dict:
            if choice >= 0:
                base_config_dict['device'] = 'cuda:' + str(choice)
                if gpu_name:
                    base_config_dict['device_name'] = gpu_name
            else:
                base_config_dict['device'] = 'cpu'
        if not auto_device and 'device' not in config_dict:
            base_config_dict['device'] = 'cpu'
        # reload hyper-parameter from parameter dict
        for key in config_dict:
            base_config_dict[key] = config_dict[key]

    if 'SRD' in base_config_dict:
        assert base_config_dict['SRD'] >= 0
    if 'lcf' in base_config_dict:
        assert base_config_dict['lcf'] in {'cdw', 'cdm', 'fusion'}
    if 'window' in base_config_dict:
        assert base_config_dict['window'] in {'l', 'r', 'lr'}
    if 'eta' in base_config_dict:
        assert base_config_dict['eta'] == -1 or 0 <= base_config_dict['eta'] <= 1
    if 'similarity_threshold' in base_config_dict:
        assert 0 <= base_config_dict['similarity_threshold'] <= 1
    if 'num_epoch' in base_config_dict:
        assert 0 <= base_config_dict['evaluate_begin'] < base_config_dict['num_epoch']
    if 'cross_validate_fold' in base_config_dict:
        assert base_config_dict['cross_validate_fold'] == -1 or 5 <= base_config_dict['cross_validate_fold'] <= 10

    base_config_dict['model_name'] = base_config_dict['model'].__name__.lower()
    base_config_dict['Version'] = __version__

    apc_config = Namespace(**base_config_dict)

    if apc_config.use_syntax_based_SRD:
        print('-' * 130)
        print('Force to use syntax distance-based semantic-relative distance,'
              ' however Chinese is not supported to parse syntax distance yet!  ')
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

    dataset_file = detect_dataset(dataset_path, auto_evaluate, task='apc_benchmark')

    config = init_config(parameter_dict, apc_config_handler.get_apc_param_dict_base(), auto_device)
    config.dataset_path = dataset_path
    config.model_path_to_save = model_path_to_save
    config.dataset_file = dataset_file
    model_path = []
    sent_classifier = None

    if hasattr(APCModelList.GloVeAPCModelList, parameter_dict['model'].__name__):
        train4apc_func = train4apc_glove
    else:
        train4apc_func = train4apc

    if isinstance(config.seed, int):
        config.seed = [config.seed]

    if os.path.exists(config.dataset_path):
        log_name = '{}_{}_srd{}_custom_dataset'.format(config.model_name, config.lcf, config.SRD)
    else:
        log_name = '{}_{}_srd{}_{}'.format(config.model_name, config.lcf, config.SRD, config.dataset_path)

    logger = get_logger(os.getcwd(), log_name=log_name, log_type='training')
    for _, s in enumerate(config.seed):
        t_config = Namespace(**vars(config))
        t_config.seed = s
        if model_path_to_save:
            model_path.append(train4apc_func(t_config, logger))
        else:
            # always return the last trained model if dont save trained models
            sent_classifier = SentimentClassifier(model_arg=train4apc_func(t_config, logger))
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    if model_path_to_save:
        return SentimentClassifier(model_arg=max(model_path))
    else:
        return sent_classifier


def load_sentiment_classifier(trained_model_path=None,
                              sentiment_map=None,
                              auto_device=True):
    infer_model = SentimentClassifier(trained_model_path, sentiment_map=sentiment_map)
    infer_model.to('cuda:' + str(choice)) if auto_device and choice >= 0 else infer_model.cpu()
    return infer_model


def train_atepc(parameter_dict=None,
                dataset_path=None,
                model_path_to_save=None,
                auto_evaluate=True,
                auto_device=True):
    '''
    evaluate model performance while training_tutorials model in order to obtain best benchmarked model
    '''

    dataset_file = detect_dataset(dataset_path, auto_evaluate, task='atepc_benchmark')

    config = init_config(parameter_dict, atepc_config_handler.get_atepc_param_dict_base(), auto_device)
    config.dataset_path = dataset_path
    config.model_path_to_save = model_path_to_save
    config.dataset_file = dataset_file
    model_path = []

    if isinstance(config.seed, int):
        config.seed = [config.seed]

    if os.path.exists(config.dataset_path):
        log_name = '{}_{}_srd{}_custom_dataset'.format(config.model_name, config.lcf, config.SRD)
    else:
        log_name = '{}_{}_srd{}_{}'.format(config.model_name, config.lcf, config.SRD, config.dataset_path)
    logger = get_logger(os.getcwd(), log_name=log_name, log_type='training')
    # always save all trained models in case of obtaining best performance
    # in different metrics among ATE and APC tasks.
    for _, s in enumerate(config.seed):
        t_config = Namespace(**vars(config))
        t_config.seed = s
        model_path.append(train4atepc(t_config, logger))
    logger.disabled = True
    return AspectExtractor(max(model_path))


def load_aspect_extractor(trained_model_path=None,
                          sentiment_map=None,
                          auto_device=True):
    aspect_extractor = AspectExtractor(trained_model_path, sentiment_map=sentiment_map)
    aspect_extractor.to('cuda:' + str(choice)) if auto_device and choice >= 0 else aspect_extractor.cpu()
    return aspect_extractor
