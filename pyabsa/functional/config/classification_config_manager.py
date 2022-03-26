# -*- coding: utf-8 -*-
# file: apc_config_manager.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
from pyabsa.functional.config.config_manager import ConfigManager
from pyabsa.core.tc.classic.__bert__.models import BERT
from pyabsa.core.tc.classic.__glove__.models import LSTM

_classification_config_template = {'model': BERT,
                                   'optimizer': "adam",
                                   'learning_rate': 0.00002,
                                   'patience': 99999,
                                   'pretrained_bert': "microsoft/mdeberta-v3-base",
                                   'cache_dataset': True,
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
                                   'cross_validate_fold': -1
                                   # split train and test datasets into 5 folds and repeat 3 training
                                   }

_classification_config_base = {'model': BERT,
                               'optimizer': "adam",
                               'learning_rate': 0.00002,
                               'pretrained_bert': "microsoft/deberta-v3-base",
                               'cache_dataset': True,
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
                               'cross_validate_fold': -1
                               # split train and test datasets into 5 folds and repeat 3 training
                               }

_classification_config_english = {'model': BERT,
                                  'optimizer': "adam",
                                  'learning_rate': 0.00002,
                                  'patience': 99999,
                                  'pretrained_bert': "microsoft/deberta-v3-base",
                                  'cache_dataset': True,
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
                                  'cross_validate_fold': -1
                                  # split train and test datasets into 5 folds and repeat 3 training
                                  }

_classification_config_multilingual = {'model': BERT,
                                       'optimizer': "adam",
                                       'learning_rate': 0.00002,
                                       'patience': 99999,
                                       'pretrained_bert': "microsoft/mdeberta-v3-base",
                                       'cache_dataset': True,
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
                                       'cross_validate_fold': -1
                                       # split train and test datasets into 5 folds and repeat 3 training
                                       }

_classification_config_chinese = {'model': BERT,
                                  'optimizer': "adam",
                                  'learning_rate': 0.00002,
                                  'patience': 99999,
                                  'cache_dataset': True,
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
                                  'cross_validate_fold': -1
                                  # split train and test datasets into 5 folds and repeat 3 training
                                  }

_classification_config_glove = {'model': LSTM,
                                'optimizer': "adam",
                                'learning_rate': 0.001,
                                'cache_dataset': True,
                                'show_metric': False,
                                'max_seq_len': 100,
                                'patience': 99999,
                                'dropout': 0.1,
                                'l2reg': 0.000001,
                                'num_epoch': 20,
                                'batch_size': 16,
                                'initializer': 'xavier_uniform_',
                                'seed': 52,
                                'embed_dim': 300,
                                'hidden_dim': 300,
                                'polarities_dim': 3,
                                'log_step': 5,
                                'hops': 3,  # valid in MemNet and RAM only
                                'evaluate_begin': 0,
                                'cross_validate_fold': -1
                                }


class ClassificationConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:  {'model': BERT,
                            'optimizer': "adam",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "roberta-base",
                            'cache_dataset':True,
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
    def set_classification_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == 'template':
                _classification_config_template.update(newitem)
            elif configType == 'base':
                _classification_config_base.update(newitem)
            elif configType == 'english':
                _classification_config_english.update(newitem)
            elif configType == 'chinese':
                _classification_config_chinese.update(newitem)
            elif configType == 'multilingual':
                _classification_config_multilingual.update(newitem)
            elif configType == 'glove':
                _classification_config_glove.update(newitem)
            else:
                raise ValueError("Wrong value of config type supplied, please use one from following type: template, base, english, chinese, multilingual, glove")
        else:
            raise TypeError("Wrong type of new config item supplied, please use dict e.g.{'NewConfig': NewValue}")

    @staticmethod
    def set_classification_config_template(newitem):
        ClassificationConfigManager.set_classification_config('template', newitem)

    @staticmethod
    def set_classification_config_base(newitem):
        ClassificationConfigManager.set_classification_config('base', newitem)

    @staticmethod
    def set_classification_config_english(newitem):
        ClassificationConfigManager.set_classification_config('english', newitem)

    @staticmethod
    def set_classification_config_chinese(newitem):
        ClassificationConfigManager.set_classification_config('chinese', newitem)

    @staticmethod
    def set_classification_config_multilingual(newitem):
        ClassificationConfigManager.set_classification_config('multilingual', newitem)

    @staticmethod
    def set_classification_config_glove(newitem):
        ClassificationConfigManager.set_classification_config('glove', newitem)

    @staticmethod
    def get_classification_config_template():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_template))

    @staticmethod
    def get_classification_config_base():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_base))

    @staticmethod
    def get_classification_config_english():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_english))

    @staticmethod
    def get_classification_config_chinese():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_chinese))

    @staticmethod
    def get_classification_config_multilingual():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_multilingual))

    @staticmethod
    def get_classification_config_glove():
        return ClassificationConfigManager(copy.deepcopy(_classification_config_glove))
