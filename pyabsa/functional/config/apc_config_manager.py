# -*- coding: utf-8 -*-
# file: apc_config_manager.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

from pyabsa.core.apc.classic.__bert__ import TNet_LF_BERT
from pyabsa.core.apc.classic.__glove__ import TNet_LF
from pyabsa.core.apc.models import APCModelList

from pyabsa.functional.config.config_manager import ConfigManager

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
_apc_config_template = {'model': APCModelList.BERT_SPC,
                        'optimizer': "",
                        'learning_rate': 0.00002,
                        'pretrained_bert': "bert-base-uncased",
                        'use_bert_spc': True,
                        'max_seq_len': 80,
                        'SRD': 3,
                        'dlcf_a': 2,  # the a in dlcf_dca_bert
                        'dca_p': 1,  # the p in dlcf_dca_bert
                        'dca_layer': 3,  # the layer in dlcf_dca_bert
                        'use_syntax_based_SRD': False,
                        'sigma': 0.3,
                        'lcf': "cdw",
                        'window': "lr",
                        'eta': -1,
                        'dropout': 0,
                        'l2reg': 0.0001,
                        'num_epoch': 10,
                        'batch_size': 16,
                        'initializer': 'xavier_uniform_',
                        'seed': {1, 2, 3},
                        'embed_dim': 768,
                        'hidden_dim': 768,
                        'polarities_dim': 3,
                        'log_step': 10,
                        'dynamic_truncate': True,
                        'srd_alignment': True,  # for srd_alignment
                        'evaluate_begin': 0,
                        'similarity_threshold': 1,  # disable same text check for different examples
                        'cross_validate_fold': -1
                        # split train and test datasets into 5 folds and repeat 3 training
                        }

_apc_config_base = {'model': APCModelList.BERT_SPC,
                    'optimizer': "adam",
                    'learning_rate': 0.00002,
                    'pretrained_bert': "bert-base-uncased",
                    'use_bert_spc': True,
                    'max_seq_len': 80,
                    'SRD': 3,
                    'dlcf_a': 2,  # the a in dlcf_dca_bert
                    'dca_p': 1,  # the p in dlcf_dca_bert
                    'dca_layer': 3,  # the layer in dlcf_dca_bert
                    'use_syntax_based_SRD': False,
                    'sigma': 0.3,
                    'lcf': "cdw",
                    'window': "lr",
                    'eta': -1,
                    'dropout': 0,
                    'l2reg': 0.0001,
                    'num_epoch': 10,
                    'batch_size': 16,
                    'initializer': 'xavier_uniform_',
                    'seed': {1, 2, 3},
                    'embed_dim': 768,
                    'hidden_dim': 768,
                    'polarities_dim': 3,
                    'log_step': 10,
                    'dynamic_truncate': True,
                    'srd_alignment': True,  # for srd_alignment
                    'evaluate_begin': 0,
                    'similarity_threshold': 1,  # disable same text check for different examples
                    'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                    }

_apc_config_english = {'model': APCModelList.BERT_SPC,
                       'optimizer': "adam",
                       'learning_rate': 0.00002,
                       'pretrained_bert': "bert-base-uncased",
                       'use_bert_spc': True,
                       'max_seq_len': 80,
                       'SRD': 3,
                       'dlcf_a': 2,  # the a in dlcf_dca_bert
                       'dca_p': 1,  # the p in dlcf_dca_bert
                       'dca_layer': 3,  # the layer in dlcf_dca_bert
                       'use_syntax_based_SRD': False,
                       'sigma': 0.3,
                       'lcf': "cdw",
                       'window': "lr",
                       'eta': -1,
                       'dropout': 0.5,
                       'l2reg': 0.0001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': {1, 2, 3},
                       'embed_dim': 768,
                       'hidden_dim': 768,
                       'polarities_dim': 3,
                       'log_step': 5,
                       'dynamic_truncate': True,
                       'srd_alignment': True,  # for srd_alignment
                       'evaluate_begin': 2,
                       'similarity_threshold': 1,  # disable same text check for different examples
                       'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                       }

_apc_config_multilingual = {'model': APCModelList.BERT_SPC,
                            'optimizer': "adam",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "bert-base-multilingual-uncased",
                            'use_bert_spc': True,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'window': "lr",
                            'eta': -1,
                            'dropout': 0.5,
                            'l2reg': 0.0001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {1, 2, 3},
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'polarities_dim': 3,
                            'log_step': 5,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 2,
                            'similarity_threshold': 1,  # disable same text check for different examples
                            'cross_validate_fold': -1
                            # split train and test datasets into 5 folds and repeat 3 training
                            }

_apc_config_chinese = {'model': APCModelList.BERT_SPC,
                       'optimizer': "adam",
                       'learning_rate': 0.00002,
                       'pretrained_bert': "bert-base-chinese",
                       'use_bert_spc': True,
                       'max_seq_len': 80,
                       'SRD': 3,
                       'dlcf_a': 2,  # the a in dlcf_dca_bert
                       'dca_p': 1,  # the p in dlcf_dca_bert
                       'dca_layer': 3,  # the layer in dlcf_dca_bert
                       'use_syntax_based_SRD': False,
                       'sigma': 0.3,
                       'lcf': "cdw",
                       'window': "lr",
                       'eta': -1,
                       'dropout': 0.5,
                       'l2reg': 0.00001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': {1, 2, 3},
                       'embed_dim': 768,
                       'hidden_dim': 768,
                       'polarities_dim': 3,
                       'log_step': 5,
                       'dynamic_truncate': True,
                       'srd_alignment': True,  # for srd_alignment
                       'evaluate_begin': 2,
                       'similarity_threshold': 1,  # disable same text check for different examples
                       'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                       }

_apc_config_glove = {'model': TNet_LF,
                     'optimizer': "adam",
                     'learning_rate': 0.001,
                     'max_seq_len': 100,
                     'dropout': 0.1,
                     'l2reg': 0.0001,
                     'num_epoch': 20,
                     'batch_size': 16,
                     'initializer': 'xavier_uniform_',
                     'seed': {1, 2, 3},
                     'embed_dim': 300,
                     'hidden_dim': 300,
                     'polarities_dim': 3,
                     'log_step': 5,
                     'hops': 3,  # valid in MemNet and RAM only
                     'evaluate_begin': 0,
                     'similarity_threshold': 1,  # disable same text check for different examples
                     'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                     }

_apc_config_bert_baseline = {'model': TNet_LF_BERT,
                             'optimizer': "adam",
                             'learning_rate': 0.00002,
                             'max_seq_len': 100,
                             'dropout': 0.1,
                             'l2reg': 0.0001,
                             'num_epoch': 5,
                             'batch_size': 16,
                             'pretrained_bert': 'bert-base-uncased',
                             'initializer': 'xavier_uniform_',
                             'seed': {1, 2, 3},
                             'embed_dim': 768,
                             'hidden_dim': 768,
                             'polarities_dim': 3,
                             'log_step': 5,
                             'hops': 3,  # valid in MemNet and RAM only
                             'evaluate_begin': 0,
                             'dynamic_truncate': True,
                             'similarity_threshold': 1,  # disable same text check for different examples
                             'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                             }


class APCConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:   {'model': APCModelList.BERT_SPC,
                            'optimizer': "",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "bert-base-uncased",
                            'use_bert_spc': True,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'window': "lr",
                            'eta': -1,
                            'dropout': 0,
                            'l2reg': 0.0001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {1, 2, 3},
                            'embed_dim': 768,
                            'hidden_dim': 768,
                            'polarities_dim': 3,
                            'log_step': 10,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 0,
                            'similarity_threshold': 1,  # disable same text check for different examples
                            'cross_validate_fold': -1   # split train and test datasets into 5 folds and repeat 3 training
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def get_apc_config_template():
        return APCConfigManager(copy.deepcopy(_apc_config_template))

    @staticmethod
    def get_apc_config_base():
        return APCConfigManager(copy.deepcopy(_apc_config_base))

    @staticmethod
    def get_apc_config_english():
        return APCConfigManager(copy.deepcopy(_apc_config_english))

    @staticmethod
    def get_apc_config_chinese():
        return APCConfigManager(copy.deepcopy(_apc_config_chinese))

    @staticmethod
    def get_apc_config_multilingual():
        return APCConfigManager(copy.deepcopy(_apc_config_multilingual))

    @staticmethod
    def get_apc_config_glove():
        return APCConfigManager(copy.deepcopy(_apc_config_glove))

    @staticmethod
    def get_apc_config_bert_baseline():
        return APCConfigManager(copy.deepcopy(_apc_config_bert_baseline))
