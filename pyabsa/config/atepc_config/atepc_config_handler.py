# -*- coding: utf-8 -*-
# file: atepc_config_handler.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.tasks.atepc.models import LCF_ATEPC

import copy

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the config use template param_dict
_atepc_param_dict_template = {'model': LCF_ATEPC,
                              'optimizer': "adamw",
                              'learning_rate': 0.00003,
                              'pretrained_bert_name': "bert-base-uncased",
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
                              'seed': {1, 2, 3},
                              'embed_dim': 768,
                              'hidden_dim': 768,
                              'polarities_dim': 2,
                              'log_step': 50,
                              'gradient_accumulation_steps': 1,
                              'dynamic_truncate': True,
                              'srd_alignment': True,  # for srd_alignment
                              'evaluate_begin': 0
                              }

_atepc_param_dict_base = {'model': LCF_ATEPC,
                          'optimizer': "adamw",
                          'learning_rate': 0.00003,
                          'pretrained_bert_name': "bert-base-uncased",
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
                          'seed': {1, 2, 3},
                          'embed_dim': 768,
                          'hidden_dim': 768,
                          'polarities_dim': 2,
                          'log_step': 50,
                          'gradient_accumulation_steps': 1,
                          'dynamic_truncate': True,
                          'srd_alignment': True,  # for srd_alignment
                          'evaluate_begin': 0
                          }

_atepc_param_dict_english = {'model': LCF_ATEPC,
                             'optimizer': "adamw",
                             'learning_rate': 0.00002,
                             'pretrained_bert_name': "bert-base-uncased",
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
                             'seed': {1, 2, 3},
                             'embed_dim': 768,
                             'hidden_dim': 768,
                             'polarities_dim': 2,
                             'log_step': 50,
                             'gradient_accumulation_steps': 1,
                             'dynamic_truncate': True,
                             'srd_alignment': True,  # for srd_alignment
                             'evaluate_begin': 0
                             }

_atepc_param_dict_chinese = {'model': LCF_ATEPC,
                             'optimizer': "adamw",
                             'learning_rate': 0.00002,
                             'pretrained_bert_name': "bert-base-chinese",
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
                             'seed': {1, 2, 3},
                             'embed_dim': 768,
                             'hidden_dim': 768,
                             'polarities_dim': 2,
                             'log_step': 50,
                             'gradient_accumulation_steps': 1,
                             'dynamic_truncate': True,
                             'srd_alignment': True,  # for srd_alignment
                             'evaluate_begin': 0
                             }

_atepc_param_dict_multilingual = {'model': LCF_ATEPC,
                                  'optimizer': "adamw",
                                  'learning_rate': 0.00002,
                                  'pretrained_bert_name': "bert-base-multilingual-uncased",
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
                                  'seed': {1, 2, 3},
                                  'embed_dim': 768,
                                  'hidden_dim': 768,
                                  'polarities_dim': 2,
                                  'log_step': 50,
                                  'gradient_accumulation_steps': 1,
                                  'dynamic_truncate': True,
                                  'srd_alignment': True,  # for srd_alignment
                                  'evaluate_begin': 0
                                  }


def get_atepc_param_dict_template():
    return copy.deepcopy(_atepc_param_dict_template)


def get_atepc_param_dict_base():
    return copy.deepcopy(_atepc_param_dict_base)


def get_atepc_param_dict_english():
    return copy.deepcopy(_atepc_param_dict_english)


def get_atepc_param_dict_chinese():
    return copy.deepcopy(_atepc_param_dict_chinese)


def get_atepc_param_dict_multilingual():
    return copy.deepcopy(_atepc_param_dict_multilingual)
