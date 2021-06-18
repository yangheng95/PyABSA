# -*- coding: utf-8 -*-
# file: apc_config.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

apc_param_dict_base = {'model_name': "slide_lcfs_bert",
                       'optimizer': "adam",
                       'learning_rate': 0.00002,
                       'pretrained_bert_name': "bert-base-uncased",
                       'use_dual_bert': False,
                       'use_bert_spc': True,
                       'max_seq_len': 50,
                       'SRD': 3,
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
                       'similarity_threshold': 0.9  # disable same text check for different examples
                       }

apc_param_dict_english = {'model_name': "slide_lcfs_bert",
                          'optimizer': "adam",
                          'learning_rate': 0.00002,
                          'pretrained_bert_name': "bert-base-uncased",
                          'use_dual_bert': False,
                          'use_bert_spc': True,
                          'max_seq_len': 80,
                          'SRD': 3,
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
                          'similarity_threshold': 0.9
                          }

apc_param_dict_multilingual = {'model_name': "slide_lcfs_bert",
                               'optimizer': "adam",
                               'learning_rate': 0.00002,
                               'pretrained_bert_name': "bert-base-multilingual-uncased",
                               'use_dual_bert': False,
                               'use_bert_spc': True,
                               'max_seq_len': 80,
                               'SRD': 3,
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
                               'similarity_threshold': 0.9
                               }

apc_param_dict_chinese = {'model_name': "slide_lcfs_bert",
                          'optimizer': "adam",
                          'learning_rate': 0.00002,
                          'pretrained_bert_name': "bert-base-chinese",
                          'use_dual_bert': False,
                          'use_bert_spc': True,
                          'max_seq_len': 80,
                          'SRD': 3,
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
                          'similarity_threshold': 0.9
                          }


def get_apc_param_dict_base():
    return copy.deepcopy(apc_param_dict_base)


def get_apc_param_dict_english():
    return copy.deepcopy(apc_param_dict_english)


def get_apc_param_dict_chinese():
    return copy.deepcopy(apc_param_dict_chinese)


def get_apc_param_dict_multilingual():
    return copy.deepcopy(apc_param_dict_multilingual)
