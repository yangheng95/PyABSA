# -*- coding: utf-8 -*-
# file: apc_config_manager.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
from pyabsa.functional.config.config_manager import ConfigManager
from pyabsa.core.tad.classic.__bert__.models import TADBERT
from pyabsa.core.tad.classic.__glove__.models import TADLSTM

_tad_config_template = {'model': TADBERT,
                        'optimizer': "adamw",
                        'learning_rate': 0.00002,
                        'patience': 99999,
                        'pretrained_bert': "microsoft/mdeberta-v3-base",
                        'cache_dataset': True,
                        'warmup_step': -1,
                        'show_metric': False,
                        'max_seq_len': 80,
                        'dropout': 0,
                        'l2reg': 0.000001,
                        'num_epoch': 10,
                        'batch_size': 16,
                        'initializer': 'xavier_uniform_',
                        'seed': 52,
                        'polarities_dim': 3,
                        'log_step': 10,
                        'evaluate_begin': 0,
                        'cross_validate_fold': -1,
                        'use_amp': False,
                        # split train and test datasets into 5 folds and repeat 3 training
                        }

_tad_config_base = {'model': TADBERT,
                    'optimizer': "adamw",
                    'learning_rate': 0.00002,
                    'pretrained_bert': "microsoft/deberta-v3-base",
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
                    'seed': 52,
                    'polarities_dim': 3,
                    'log_step': 10,
                    'evaluate_begin': 0,
                    'cross_validate_fold': -1,
                    'use_amp': False,
                    # split train and test datasets into 5 folds and repeat 3 training
                    }

_tad_config_english = {'model': TADBERT,
                       'optimizer': "adamw",
                       'learning_rate': 0.00002,
                       'patience': 99999,
                       'pretrained_bert': "microsoft/deberta-v3-base",
                       'cache_dataset': True,
                       'warmup_step': -1,
                       'show_metric': False,
                       'max_seq_len': 80,
                       'dropout': 0,
                       'l2reg': 0.000001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': 52,
                       'polarities_dim': 3,
                       'log_step': 10,
                       'evaluate_begin': 0,
                       'cross_validate_fold': -1,
                       'use_amp': False,
                       # split train and test datasets into 5 folds and repeat 3 training
                       }

_tad_config_multilingual = {'model': TADBERT,
                            'optimizer': "adamw",
                            'learning_rate': 0.00002,
                            'patience': 99999,
                            'pretrained_bert': "microsoft/mdeberta-v3-base",
                            'cache_dataset': True,
                            'warmup_step': -1,
                            'show_metric': False,
                            'max_seq_len': 80,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': 52,
                            'polarities_dim': 3,
                            'log_step': 10,
                            'evaluate_begin': 0,
                            'cross_validate_fold': -1,
                            'use_amp': False,
                            # split train and test datasets into 5 folds and repeat 3 training
                            }

_tad_config_chinese = {'model': TADBERT,
                       'optimizer': "adamw",
                       'learning_rate': 0.00002,
                       'patience': 99999,
                       'cache_dataset': True,
                       'warmup_step': -1,
                       'show_metric': False,
                       'pretrained_bert': "bert-base-chinese",
                       'max_seq_len': 80,
                       'dropout': 0,
                       'l2reg': 0.000001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': 52,
                       'polarities_dim': 3,
                       'log_step': 10,
                       'evaluate_begin': 0,
                       'cross_validate_fold': -1,
                       'use_amp': False,
                       # split train and test datasets into 5 folds and repeat 3 training
                       }

_tad_config_glove = {'model': TADLSTM,
                     'optimizer': "adamw",
                     'learning_rate': 0.001,
                     'cache_dataset': True,
                     'warmup_step': -1,
                     'show_metric': False,
                     'max_seq_len': 100,
                     'patience': 20,
                     'dropout': 0.1,
                     'l2reg': 0.000001,
                     'num_epoch': 100,
                     'batch_size': 64,
                     'initializer': 'xavier_uniform_',
                     'seed': 52,
                     'embed_dim': 300,
                     'hidden_dim': 300,
                     'polarities_dim': 3,
                     'log_step': 5,
                     'hops': 3,  # valid in MemNet and RAM only
                     'evaluate_begin': 0,
                     'cross_validate_fold': -1,
                     'use_amp': False,
                     }


class TADConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:  {'model': BERT,
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
                            'polarities_dim': 3,
                            'log_step': 10,
                            'evaluate_begin': 0,
                            'cross_validate_fold': -1 # split train and test datasets into 5 folds and repeat 3 training
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_tad_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == 'template':
                _tad_config_template.update(newitem)
            elif configType == 'base':
                _tad_config_base.update(newitem)
            elif configType == 'english':
                _tad_config_english.update(newitem)
            elif configType == 'chinese':
                _tad_config_chinese.update(newitem)
            elif configType == 'multilingual':
                _tad_config_multilingual.update(newitem)
            elif configType == 'glove':
                _tad_config_glove.update(newitem)
            else:
                raise ValueError(
                    "Wrong value of config type supplied, please use one from following type: template, base, english, chinese, multilingual, glove")
        else:
            raise TypeError("Wrong type of new config item supplied, please use dict e.g.{'NewConfig': NewValue}")

    @staticmethod
    def set_tad_config_template(newitem):
        TADConfigManager.set_tad_config('template', newitem)

    @staticmethod
    def set_tad_config_base(newitem):
        TADConfigManager.set_tad_config('base', newitem)

    @staticmethod
    def set_tad_config_english(newitem):
        TADConfigManager.set_tad_config('english', newitem)

    @staticmethod
    def set_tad_config_chinese(newitem):
        TADConfigManager.set_tad_config('chinese', newitem)

    @staticmethod
    def set_tad_config_multilingual(newitem):
        TADConfigManager.set_tad_config('multilingual', newitem)

    @staticmethod
    def set_tad_config_glove(newitem):
        TADConfigManager.set_tad_config('glove', newitem)

    @staticmethod
    def get_tad_config_template() -> ConfigManager:
        _tad_config_template.update(_tad_config_template)
        return TADConfigManager(copy.deepcopy(_tad_config_template))

    @staticmethod
    def get_tad_config_base() -> ConfigManager:
        _tad_config_template.update(_tad_config_base)
        return TADConfigManager(copy.deepcopy(_tad_config_template))

    @staticmethod
    def get_tad_config_english() -> ConfigManager:
        _tad_config_template.update(_tad_config_english)
        return TADConfigManager(copy.deepcopy(_tad_config_template))

    @staticmethod
    def get_tad_config_chinese() -> ConfigManager:
        _tad_config_template.update(_tad_config_chinese)
        return TADConfigManager(copy.deepcopy(_tad_config_template))

    @staticmethod
    def get_tad_config_multilingual() -> ConfigManager:
        _tad_config_template.update(_tad_config_multilingual)
        return TADConfigManager(copy.deepcopy(_tad_config_template))

    @staticmethod
    def get_tad_config_glove() -> ConfigManager:
        _tad_config_template.update(_tad_config_glove)
        return TADConfigManager(copy.deepcopy(_tad_config_template))
