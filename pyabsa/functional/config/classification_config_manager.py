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
                                   'pretrained_bert': "bert-base-uncased",
                                   'max_seq_len': 80,
                                   'dropout': 0,
                                   'l2reg': 0.0001,
                                   'num_epoch': 10,
                                   'batch_size': 16,
                                   'initializer': 'xavier_uniform_',
                                   'seed': 52,
                                   'embed_dim': 768,
                                   'hidden_dim': 768,
                                   'polarities_dim': 3,
                                   'log_step': 10,
                                   'evaluate_begin': 0,
                                   'cross_validate_fold': -1
                                   # split train and test datasets into 5 folds and repeat 3 training
                                   }

_classification_config_base = {'model': BERT,
                               'optimizer': "adam",
                               'learning_rate': 0.00002,
                               'pretrained_bert': "bert-base-uncased",
                               'max_seq_len': 80,
                               'dropout': 0,
                               'l2reg': 0.0001,
                               'num_epoch': 10,
                               'batch_size': 16,
                               'initializer': 'xavier_uniform_',
                               'seed': 52,
                               'embed_dim': 768,
                               'hidden_dim': 768,
                               'polarities_dim': 3,
                               'log_step': 10,
                               'evaluate_begin': 0,
                               'cross_validate_fold': -1
                               # split train and test datasets into 5 folds and repeat 3 training
                               }

_classification_config_english = {'model': BERT,
                                  'optimizer': "adam",
                                  'learning_rate': 0.00002,
                                  'pretrained_bert': "bert-base-uncased",
                                  'max_seq_len': 80,
                                  'dropout': 0,
                                  'l2reg': 0.0001,
                                  'num_epoch': 10,
                                  'batch_size': 16,
                                  'initializer': 'xavier_uniform_',
                                  'seed': 52,
                                  'embed_dim': 768,
                                  'hidden_dim': 768,
                                  'polarities_dim': 3,
                                  'log_step': 10,
                                  'evaluate_begin': 0,
                                  'cross_validate_fold': -1
                                  # split train and test datasets into 5 folds and repeat 3 training
                                  }

_classification_config_multilingual = {'model': BERT,
                                       'optimizer': "adam",
                                       'learning_rate': 0.00002,
                                       'pretrained_bert': "bert-base-multilingual-uncased",
                                       'max_seq_len': 80,
                                       'dropout': 0,
                                       'l2reg': 0.0001,
                                       'num_epoch': 10,
                                       'batch_size': 16,
                                       'initializer': 'xavier_uniform_',
                                       'seed': 52,
                                       'embed_dim': 768,
                                       'hidden_dim': 768,
                                       'polarities_dim': 3,
                                       'log_step': 10,
                                       'evaluate_begin': 0,
                                       'cross_validate_fold': -1
                                       # split train and test datasets into 5 folds and repeat 3 training
                                       }

_classification_config_chinese = {'model': BERT,
                                  'optimizer': "adam",
                                  'learning_rate': 0.00002,
                                  'pretrained_bert': "bert-base-chinese",
                                  'max_seq_len': 80,
                                  'dropout': 0,
                                  'l2reg': 0.0001,
                                  'num_epoch': 10,
                                  'batch_size': 16,
                                  'initializer': 'xavier_uniform_',
                                  'seed': 52,
                                  'embed_dim': 768,
                                  'hidden_dim': 768,
                                  'polarities_dim': 3,
                                  'log_step': 10,
                                  'evaluate_begin': 0,
                                  'cross_validate_fold': -1
                                  # split train and test datasets into 5 folds and repeat 3 training
                                  }

_classification_config_glove = {'model': LSTM,
                                'optimizer': "adam",
                                'learning_rate': 0.001,
                                'max_seq_len': 100,
                                'dropout': 0.1,
                                'l2reg': 0.0001,
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
                            'pretrained_bert': "bert-base-uncased",
                            'max_seq_len': 80,
                            'dropout': 0,
                            'l2reg': 0.0001,
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
