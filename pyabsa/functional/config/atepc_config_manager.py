# -*- coding: utf-8 -*-
# file: apc_config_manager.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

from pyabsa.functional.config.config_manager import ConfigManager
from pyabsa.core.atepc.models.lcf_atepc import LCF_ATEPC

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
_atepc_config_template = {'model': LCF_ATEPC,
                          'optimizer': "adamw",
                          'learning_rate': 0.00003,
                          'pretrained_bert': "microsoft/deberta-v3-base",
                          'cache_dataset': True,
                          'warmup_step': -1,
                          'use_bert_spc': False,
                          'show_metric': False,
                          'max_seq_len': 80,
                          'SRD': 3,
                          'use_syntax_based_SRD': False,
                          'lcf': "cdw",
                          'window': "lr",  # unused yet
                          'dropout': 0.5,
                          'l2reg': 0.000001,
                          'num_epoch': 10,
                          'batch_size': 16,
                          'initializer': 'xavier_uniform_',
                          'seed': 52,
                          'polarities_dim': 2,
                          'log_step': 50,
                          'patience': 99999,
                          'gradient_accumulation_steps': 1,
                          'dynamic_truncate': True,
                          'srd_alignment': True,  # for srd_alignment
                          'evaluate_begin': 0
                          }

_atepc_config_base = {'model': LCF_ATEPC,
                      'optimizer': "adamw",
                      'learning_rate': 0.00003,
                      'pretrained_bert': "microsoft/deberta-v3-base",
                      'cache_dataset': True,
                      'warmup_step': -1,
                      'use_bert_spc': False,
                      'show_metric': False,
                      'max_seq_len': 80,
                      'patience': 5,
                      'SRD': 3,
                      'use_syntax_based_SRD': False,
                      'lcf': "cdw",
                      'window': "lr",  # unused yet
                      'dropout': 0.5,
                      'l2reg': 0.000001,
                      'num_epoch': 10,
                      'batch_size': 16,
                      'initializer': 'xavier_uniform_',
                      'seed': 52,
                      'polarities_dim': 2,
                      'log_step': 50,
                      'gradient_accumulation_steps': 1,
                      'dynamic_truncate': True,
                      'srd_alignment': True,  # for srd_alignment
                      'evaluate_begin': 0
                      }

_atepc_config_english = {'model': LCF_ATEPC,
                         'optimizer': "adamw",
                         'learning_rate': 0.00002,
                         'pretrained_bert': "microsoft/deberta-v3-base",
                         'cache_dataset': True,
                         'warmup_step': -1,
                         'use_bert_spc': False,
                         'show_metric': False,
                         'max_seq_len': 80,
                         'SRD': 3,
                         'use_syntax_based_SRD': False,
                         'lcf': "cdw",
                         'window': "lr",
                         'dropout': 0.5,
                         'l2reg': 0.00001,
                         'num_epoch': 10,
                         'batch_size': 16,
                         'initializer': 'xavier_uniform_',
                         'seed': 52,
                         'polarities_dim': 2,
                         'log_step': 50,
                         'patience': 99999,
                         'gradient_accumulation_steps': 1,
                         'dynamic_truncate': True,
                         'srd_alignment': True,  # for srd_alignment
                         'evaluate_begin': 0
                         }

_atepc_config_chinese = {'model': LCF_ATEPC,
                         'optimizer': "adamw",
                         'learning_rate': 0.00002,
                         'pretrained_bert': "bert-base-chinese",
                         'use_bert_spc': False,
                         'cache_dataset': True,
                         'warmup_step': -1,
                         'show_metric': False,
                         'max_seq_len': 80,
                         'SRD': 3,
                         'use_syntax_based_SRD': False,
                         'lcf': "cdw",
                         'window': "lr",
                         'dropout': 0.5,
                         'l2reg': 0.00001,
                         'num_epoch': 10,
                         'batch_size': 16,
                         'initializer': 'xavier_uniform_',
                         'seed': 52,
                         'polarities_dim': 2,
                         'log_step': 50,
                         'patience': 99999,
                         'gradient_accumulation_steps': 1,
                         'dynamic_truncate': True,
                         'srd_alignment': True,  # for srd_alignment
                         'evaluate_begin': 0
                         }

_atepc_config_multilingual = {'model': LCF_ATEPC,
                              'optimizer': "adamw",
                              'learning_rate': 0.00002,
                              'pretrained_bert': "bert-base-multilingual-uncased",
                              'use_bert_spc': False,
                              'cache_dataset': True,
                              'warmup_step': -1,
                              'show_metric': False,
                              'max_seq_len': 80,
                              'SRD': 3,
                              'use_syntax_based_SRD': False,
                              'lcf': "cdw",
                              'window': "lr",
                              'dropout': 0.5,
                              'l2reg': 0.00001,
                              'num_epoch': 10,
                              'batch_size': 16,
                              'initializer': 'xavier_uniform_',
                              'seed': 52,
                              'polarities_dim': 2,
                              'log_step': 50,
                              'patience': 99999,
                              'gradient_accumulation_steps': 1,
                              'dynamic_truncate': True,
                              'srd_alignment': True,  # for srd_alignment
                              'evaluate_begin': 0
                              }


class ATEPCConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params: {'model': LCF_ATEPC,
                          'optimizer': "adamw",
                          'learning_rate': 0.00003,
                          'pretrained_bert': "microsoft/deberta-v3-base",
                          'cache_dataset': True,
                          'warmup_step': -1,
                          'use_bert_spc': False,
                          'show_metric': False,
                          'max_seq_len': 80,
                          'SRD': 3,
                          'use_syntax_based_SRD': False,
                          'lcf': "cdw",
                          'window': "lr",  # unused yet
                          'dropout': 0.5,
                          'l2reg': 0.000001,
                          'num_epoch': 10,
                          'batch_size': 16,
                          'initializer': 'xavier_uniform_',
                          'seed': {52, 512, 2}
                          'polarities_dim': 2,
                          'log_step': 50,
                          'patience': 99999,
                          'gradient_accumulation_steps': 1,
                          'dynamic_truncate': True,
                          'srd_alignment': True,  # for srd_alignment
                          'evaluate_begin': 0
                          }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_atepc_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == 'template':
                _atepc_config_template.update(newitem)
            elif configType == 'base':
                _atepc_config_base.update(newitem)
            elif configType == 'english':
                _atepc_config_english.update(newitem)
            elif configType == 'chinese':
                _atepc_config_chinese.update(newitem)
            elif configType == 'multilingual':
                _atepc_config_multilingual.update(newitem)
            else:
                raise ValueError("Wrong value of config type supplied, please use one from following type: template, base, english, chinese, multilingual")
        else:
            raise TypeError("Wrong type of new config item supplied, please use dict e.g.{'NewConfig': NewValue}")

    @staticmethod
    def set_atepc_config_template(newitem):
        ATEPCConfigManager.set_atepc_config('template', newitem)

    @staticmethod
    def set_atepc_config_base(newitem):
        ATEPCConfigManager.set_atepc_config('base', newitem)

    @staticmethod
    def set_atepc_config_english(newitem):
        ATEPCConfigManager.set_atepc_config('english', newitem)

    @staticmethod
    def set_atepc_config_chinese(newitem):
        ATEPCConfigManager.set_atepc_config('chinese', newitem)

    @staticmethod
    def set_atepc_config_multilingual(newitem):
        ATEPCConfigManager.set_atepc_config('multilingual', newitem)

    @staticmethod
    def get_atepc_config_template() -> ConfigManager:
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_base() -> ConfigManager:
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_base))

    @staticmethod
    def get_atepc_config_english() -> ConfigManager:
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_english))

    @staticmethod
    def get_atepc_config_chinese() -> ConfigManager:
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_chinese))

    @staticmethod
    def get_atepc_config_multilingual() -> ConfigManager:
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_multilingual))
