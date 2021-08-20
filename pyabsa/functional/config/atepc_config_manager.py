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
                          'pretrained_bert': "bert-base-uncased",
                          'use_bert_spc': False,
                          'max_seq_len': 80,
                          'SRD': 3,
                          'use_syntax_based_SRD': False,
                          'lcf': "cdw",
                          'window': "lr",  # unused yet
                          'dropout': 0.5,
                          'l2reg': 0.0001,
                          'num_epoch': 10,
                          'batch_size': 16,
                          'initializer': 'xavier_uniform_',
                          'seed': 52,
                          'embed_dim': 768,
                          'hidden_dim': 768,
                          'polarities_dim': 2,
                          'log_step': 50,
                          'gradient_accumulation_steps': 1,
                          'dynamic_truncate': True,
                          'srd_alignment': True,  # for srd_alignment
                          'evaluate_begin': 0
                          }

_atepc_config_base = {'model': LCF_ATEPC,
                      'optimizer': "adamw",
                      'learning_rate': 0.00003,
                      'pretrained_bert': "bert-base-uncased",
                      'use_bert_spc': False,
                      'max_seq_len': 80,
                      'SRD': 3,
                      'use_syntax_based_SRD': False,
                      'lcf': "cdw",
                      'window': "lr",  # unused yet
                      'dropout': 0.5,
                      'l2reg': 0.0001,
                      'num_epoch': 10,
                      'batch_size': 16,
                      'initializer': 'xavier_uniform_',
                      'seed': 52,
                      'embed_dim': 768,
                      'hidden_dim': 768,
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
                         'pretrained_bert': "bert-base-uncased",
                         'use_bert_spc': False,
                         'max_seq_len': 80,
                         'SRD': 3,
                         'use_syntax_based_SRD': False,
                         'lcf': "cdw",
                         'window': "lr",
                         'dropout': 0.5,
                         'l2reg': 0.00005,
                         'num_epoch': 10,
                         'batch_size': 16,
                         'initializer': 'xavier_uniform_',
                         'seed': 52,
                         'embed_dim': 768,
                         'hidden_dim': 768,
                         'polarities_dim': 2,
                         'log_step': 50,
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
                         'max_seq_len': 80,
                         'SRD': 3,
                         'use_syntax_based_SRD': False,
                         'lcf': "cdw",
                         'window': "lr",
                         'dropout': 0.5,
                         'l2reg': 0.00005,
                         'num_epoch': 10,
                         'batch_size': 16,
                         'initializer': 'xavier_uniform_',
                         'seed': 52,
                         'embed_dim': 768,
                         'hidden_dim': 768,
                         'polarities_dim': 2,
                         'log_step': 50,
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
                              'max_seq_len': 80,
                              'SRD': 3,
                              'use_syntax_based_SRD': False,
                              'lcf': "cdw",
                              'window': "lr",
                              'dropout': 0.5,
                              'l2reg': 0.00005,
                              'num_epoch': 10,
                              'batch_size': 16,
                              'initializer': 'xavier_uniform_',
                              'seed': 52,
                              'embed_dim': 768,
                              'hidden_dim': 768,
                              'polarities_dim': 2,
                              'log_step': 50,
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
                          'pretrained_bert': "bert-base-uncased",
                          'use_bert_spc': False,
                          'max_seq_len': 80,
                          'SRD': 3,
                          'use_syntax_based_SRD': False,
                          'lcf': "cdw",
                          'window': "lr",  # unused yet
                          'dropout': 0.5,
                          'l2reg': 0.0001,
                          'num_epoch': 10,
                          'batch_size': 16,
                          'initializer': 'xavier_uniform_',
                          'seed': {52, 512, 2}
                          'embed_dim': 768,
                          'hidden_dim': 768,
                          'polarities_dim': 2,
                          'log_step': 50,
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
    def get_atepc_config_template():
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_template))

    @staticmethod
    def get_atepc_config_base():
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_base))

    @staticmethod
    def get_atepc_config_english():
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_english))

    @staticmethod
    def get_atepc_config_chinese():
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_chinese))

    @staticmethod
    def get_atepc_config_multilingual():
        return ATEPCConfigManager(copy.deepcopy(_atepc_config_multilingual))
