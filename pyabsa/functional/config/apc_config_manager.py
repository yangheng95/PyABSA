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
                        'pretrained_bert': "microsoft/deberta-v3-base",
                        'cache_dataset': True,
                        'deep_ensemble': False,
                        'use_bert_spc': True,
                        'show_metric': False,
                        'max_seq_len': 80,
                        'patience': 99999,
                        'SRD': 3,
                        'dlcf_a': 2,  # the a in dlcf_dca_bert
                        'dca_p': 1,  # the p in dlcf_dca_bert
                        'dca_layer': 3,  # the layer in dlcf_dca_bert
                        'use_syntax_based_SRD': False,
                        'sigma': 0.3,
                        'lcf': "cdw",
                        'lsa': False,
                        'window': "lr",
                        'eta': -1,
                        'dropout': 0,
                        'l2reg': 0.000001,
                        'num_epoch': 10,
                        'batch_size': 16,
                        'initializer': 'xavier_uniform_',
                        'seed': 52,
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
                    'pretrained_bert': "microsoft/deberta-v3-base",
                    'cache_dataset': True,
                    'deep_ensemble': False,
                    'show_metric': False,
                    'patience': 5,
                    'use_bert_spc': True,
                    'max_seq_len': 80,
                    'SRD': 3,
                    'dlcf_a': 2,  # the a in dlcf_dca_bert
                    'dca_p': 1,  # the p in dlcf_dca_bert
                    'dca_layer': 3,  # the layer in dlcf_dca_bert
                    'use_syntax_based_SRD': False,
                    'sigma': 0.3,
                    'lcf': "cdw",
                    'lsa': False,
                    'window': "lr",
                    'eta': -1,
                    'dropout': 0,
                    'l2reg': 0.000001,
                    'num_epoch': 10,
                    'batch_size': 16,
                    'initializer': 'xavier_uniform_',
                    'seed': 52,
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
                       'pretrained_bert': "microsoft/deberta-v3-base",
                       'cache_dataset': True,
                       'deep_ensemble': False,
                       'show_metric': False,
                       'patience': 99999,
                       'use_bert_spc': True,
                       'max_seq_len': 80,
                       'SRD': 3,
                       'dlcf_a': 2,  # the a in dlcf_dca_bert
                       'dca_p': 1,  # the p in dlcf_dca_bert
                       'dca_layer': 3,  # the layer in dlcf_dca_bert
                       'use_syntax_based_SRD': False,
                       'sigma': 0.3,
                       'lcf': "cdw",
                       'lsa': False,
                       'window': "lr",
                       'eta': -1,
                       'dropout': 0.5,
                       'l2reg': 0.000001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': 52,
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
                            'pretrained_bert': "microsoft/mdeberta-v3-base",
                            'use_bert_spc': True,
                            'cache_dataset': True,
                            'deep_ensemble': False,
                            'show_metric': False,
                            'patience': 99999,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'lsa': False,
                            'window': "lr",
                            'eta': -1,
                            'dropout': 0.5,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': 52,
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
                       'cache_dataset': True,
                       'deep_ensemble': False,
                       'show_metric': False,
                       'patience': 99999,
                       'max_seq_len': 80,
                       'SRD': 3,
                       'dlcf_a': 2,  # the a in dlcf_dca_bert
                       'dca_p': 1,  # the p in dlcf_dca_bert
                       'dca_layer': 3,  # the layer in dlcf_dca_bert
                       'use_syntax_based_SRD': False,
                       'sigma': 0.3,
                       'lcf': "cdw",
                       'lsa': False,
                       'window': "lr",
                       'eta': -1,
                       'dropout': 0.5,
                       'l2reg': 0.00001,
                       'num_epoch': 10,
                       'batch_size': 16,
                       'initializer': 'xavier_uniform_',
                       'seed': 52,
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
                     'cache_dataset': True,
                     'deep_ensemble': False,
                     'show_metric': False,
                     'patience': 5,
                     'lsa': False,
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
                     'similarity_threshold': 1,  # disable same text check for different examples
                     'cross_validate_fold': -1  # split train and test datasets into 5 folds and repeat 3 training
                     }

_apc_config_bert_baseline = {'model': TNet_LF_BERT,
                             'optimizer': "adam",
                             'learning_rate': 0.00002,
                             'max_seq_len': 100,
                             'deep_ensemble': False,
                             'show_metric': False,
                             'cache_dataset': True,
                             'patience': 99999,
                             'dropout': 0.1,
                             'lsa': False,
                             'l2reg': 0.000001,
                             'num_epoch': 5,
                             'batch_size': 16,
                             'pretrained_bert': 'bert-base-uncased',
                             'initializer': 'xavier_uniform_',
                             'seed': 52,
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
                            'pretrained_bert': "microsoft/deberta-v3-base",
                            'cache_dataset': True,
                            'deep_ensemble': False,
                            'show_metric': False,
                            'patience': 99999,
                            'use_bert_spc': True,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'lsa': False,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'window': "lr",
                            'eta': -1,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {52, 214}
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
    def set_apc_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == 'template':
                _apc_config_template.update(newitem)
            elif configType == 'base':
                _apc_config_base.update(newitem)
            elif configType == 'english':
                _apc_config_english.update(newitem)
            elif configType == 'chinese':
                _apc_config_chinese.update(newitem)
            elif configType == 'multilingual':
                _apc_config_multilingual.update(newitem)
            elif configType == 'glove':
                _apc_config_glove.update(newitem)
            elif configType == 'bert_baseline':
                _apc_config_bert_baseline.update(newitem)
            else:
                raise ValueError("Wrong value of config type supplied, please use one from following type: template, base, english, chinese, multilingual, glove, bert_baseline")
        else:
            raise TypeError("Wrong type of new config item supplied, please use dict e.g.{'NewConfig': NewValue}")

    @staticmethod
    def set_apc_config_template(newitem):
        APCConfigManager.set_apc_config('template', newitem)

    @staticmethod
    def set_apc_config_base(newitem):
        APCConfigManager.set_apc_config('base', newitem)

    @staticmethod
    def set_apc_config_english(newitem):
        APCConfigManager.set_apc_config('english', newitem)

    @staticmethod
    def set_apc_config_chinese(newitem):
        APCConfigManager.set_apc_config('chinese', newitem)

    @staticmethod
    def set_apc_config_multilingual(newitem):
        APCConfigManager.set_apc_config('multilingual', newitem)

    @staticmethod
    def set_apc_config_glove(newitem):
        APCConfigManager.set_apc_config('glove', newitem)

    @staticmethod
    def set_apc_config_bert_baseline(newitem):
        APCConfigManager.set_apc_config('bert_baseline', newitem)

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
