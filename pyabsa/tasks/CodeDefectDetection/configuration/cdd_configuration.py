# -*- coding: utf-8 -*-
# file: cdd_configuration.py
# time: 02/11/2022 19:57
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import copy

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
from pyabsa.framework.configuration_class.configuration_template import ConfigManager
from ..models.__classic__.lstm import LSTM
from ..models.__plm__.bert import BERT_MLP

_cdd_config_base = {'model': BERT_MLP,
                    'optimizer': "adamw",
                    'learning_rate': 0.00002,
                    'pretrained_bert': "Salesforce/codet5-small",
                    'cache_dataset': True,
                    'warmup_step': -1,
                    'show_metric': True,
                    'use_amp': False,
                    'max_seq_len': 512,
                    'patience': 99999,
                    'dropout': 0,
                    'l2reg': 0.000001,
                    'num_epoch': 10,
                    'batch_size': 16,
                    'initializer': 'xavier_uniform_',
                    'seed': 52,
                    'output_dim': 2,
                    'log_step': 10,
                    'evaluate_begin': 0,
                    'cross_validate_fold': -1
                    # split train and test datasets into 5 folds and repeat 3 trainer
                    }


class CDDConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:  {'model': MLP,
                            'optimizer': "adamw",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "roberta-base",
                            'cache_dataset': True,
                            'warmup_step': -1,
                            'show_metric': False,
                            'max_seq_len': 80,
                            'patience': 99999,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {52, 25}
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'output_dim': 3,
                            'log_step': 10,
                            'evaluate_begin': 0,
                            'cross_validate_fold': -1 # split train and test datasets into 5 folds and repeat 3 trainer
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_cdd_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == 'base':
                _cdd_config_base.update(newitem)
            else:
                raise ValueError(
                    "Wrong value of configuration_class type supplied, please use one from following type:  base")
        else:
            raise TypeError(
                "Wrong type of new configuration_class item supplied, please use dict e.g.{'NewConfig': NewValue}")

    @staticmethod
    def set_cdd_config_base(newitem):
        CDDConfigManager.set_cdd_config('base', newitem)

    @staticmethod
    def get_cdd_config_base():
        return CDDConfigManager(copy.deepcopy(_cdd_config_base))
