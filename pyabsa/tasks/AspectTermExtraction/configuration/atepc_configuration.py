# -*- coding: utf-8 -*-
# file: atepc_configuration.py
# time: 02/11/2022 19:54
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import copy

from pyabsa.framework.configuration_class.configuration_template import ConfigManager
from pyabsa.tasks.AspectTermExtraction.models.__lcf__.lcf_atepc import LCF_ATEPC

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
_atepc_config_template = {'model': LCF_ATEPC,
                          'optimizer': "adamw",
                          'learning_rate': 0.00003,
                          'cache_dataset': True,
                          'warmup_step': -1,
                          'use_bert_spc': True,
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
                          'output_dim': 2,
                          'log_step': 50,
                          'patience': 99999,
                          'gradient_accumulation_steps': 1,
                          'dynamic_truncate': True,
                          'srd_alignment': True,  # for srd_alignment
                          'evaluate_begin': 0,
                          'use_amp': False,
                          'overwrite_cache': False,

                          }

_atepc_config_base = {'model': LCF_ATEPC,
                      'optimizer': "adamw",
                      'learning_rate': 0.00003,
                      'pretrained_bert': "microsoft/deberta-v3-base",
                      'cache_dataset': True,
                      'warmup_step': -1,
                      'use_bert_spc': True,
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
                      'output_dim': 2,
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
                         'use_bert_spc': True,
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
                         'output_dim': 2,
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
                         'use_bert_spc': True,
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
                         'output_dim': 2,
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
                              'use_bert_spc': True,
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
                              'output_dim': 2,
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
                          'use_bert_spc': True,
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
                          'output_dim': 2,
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
                raise ValueError(
                    "Wrong value of configuration_class type supplied, please use one from following type: template, base, english, chinese, multilingual")
        else:
            raise TypeError("Wrong type of new configuration_class item supplied, please use dict e.g.{'NewConfig': NewValue}")

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
    def get_atepc_config_template():
        _atepc_config_template.update(_atepc_config_template)
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_base():
        _atepc_config_template.update(_atepc_config_base)
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_english():
        _atepc_config_template.update(_atepc_config_english)
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_chinese():
        _atepc_config_template.update(_atepc_config_chinese)
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_multilingual():
        _atepc_config_template.update(_atepc_config_multilingual)
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))
