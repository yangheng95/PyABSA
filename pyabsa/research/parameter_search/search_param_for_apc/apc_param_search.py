# -*- coding: utf-8 -*-
# file: atepc_param_search.py
# time: 2021/6/13 0013
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

from pyabsa import ABSADatasets, detect_dataset
from pyabsa.functional import init_config

from pyabsa.config.apc_config import apc_config_handler

from pyabsa.tasks.apc.training.apc_trainer import train4apc

from pyabsa.utils.logger import get_logger


def apc_param_search(parameter_dict=None,
                     dataset_path=None,
                     search_param: list = None,
                     auto_evaluate=True,
                     auto_device=True):
    logger = get_logger('param_search_logs', 'param_search_{}'.format(search_param[0]), 'optimal_param')

    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]
        logger.info('Search on Datasets: {}'.format(dataset_path))
        logger.info('Search with random seed: {1, 2, 3}')
        logger.info('Alternative param to test:')
        logger.info(search_param)

    optimal_param = 'None'
    max_score = 0
    max_acc = 0
    max_f1 = 0
    output = {}
    for alternative in search_param[1]:
        parameter_dict[search_param[0]] = alternative
        logger.info('*********************** Set {} = {} ***********************'.format(search_param[0], alternative))
        score = 0
        for dataset in dataset_path:
            dataset_file = detect_dataset(dataset, auto_evaluate, task='apc_benchmark')
            config = init_config(parameter_dict, apc_config_handler.get_apc_param_dict_base(), auto_device)
            config.logger = logger
            config.dataset_path = dataset
            config.model_path_to_save = None
            config.dataset_file = dataset_file
            config.dataset_path = dataset
            config.seed = [1, 2, 3]
            for _, s in enumerate(config.seed):
                t_config = copy.deepcopy(config)
                t_config.seed = s
                # if order_by == 'acc':
                #     score += train4apc(t_config)[3]
                # else:
                #     score += train4apc(t_config)[4]
                running_result = train4apc(t_config)
                t_acc, t_f1 = running_result[3], running_result[4]
                score += (t_acc + t_f1)
                if t_acc > max_acc:
                    output['max_acc'] = 'Max acc recorded while {} is {}'.format(search_param[0], alternative)
                    max_acc = t_acc
                if t_f1 > max_f1:
                    output['max_f1'] = 'Max F1 recorded while {} is {}'.format(search_param[0], alternative)
                    max_f1 = t_f1
        output[alternative] = '{}: {} tested on dataset {} scored: {}'.format(search_param[0],
                                                                              alternative,
                                                                              ','.join(dataset_path),
                                                                              score)
        print()
        if score > max_score:
            max_score = score
            optimal_param = alternative
    output['optimal'] = 'Optimal {} tested on dataset(s) {} is {}'.format(search_param[0],
                                                                          ','.join(dataset_path),
                                                                          optimal_param)
    logger.info('************************* Param Search Report *************************')
    logger.info(output['optimal'])
    for alternative in search_param[1]:
        logger.info(output[alternative])
    logger.info(output['max_acc'])
    logger.info(output['max_f1'])
    logger.info('************************* Param Search Report *************************')
    print()
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])
    return optimal_param
