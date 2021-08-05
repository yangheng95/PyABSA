# -*- coding: utf-8 -*-
# file: classification_config_handler.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import copy

from pyabsa.tasks.text_classification.__bert__.models import BERT
from pyabsa.tasks.text_classification.__glove__.models import LSTM

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the config use template param_dict
_classification_param_dict_template = {'model': BERT,
                                       'optimizer': "adam",
                                       'learning_rate': 0.00002,
                                       'pretrained_bert_name': "bert-base-uncased",
                                       'max_seq_len': 80,
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
                                       'evaluate_begin': 0,
                                       'cross_validate_fold': -1
                                       # split train and test datasets into 5 folds and repeat 3 training
                                       }

_classification_param_dict_base = {'model': BERT,
                                   'optimizer': "adam",
                                   'learning_rate': 0.00002,
                                   'pretrained_bert_name': "bert-base-uncased",
                                   'max_seq_len': 80,
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
                                   'evaluate_begin': 0,
                                   'cross_validate_fold': -1
                                   # split train and test datasets into 5 folds and repeat 3 training
                                   }

_classification_param_dict_english = {'model': BERT,
                                      'optimizer': "adam",
                                      'learning_rate': 0.00002,
                                      'pretrained_bert_name': "bert-base-uncased",
                                      'max_seq_len': 80,
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
                                      'evaluate_begin': 0,
                                      'cross_validate_fold': -1
                                      # split train and test datasets into 5 folds and repeat 3 training
                                      }

_classification_param_dict_multilingual = {'model': BERT,
                                           'optimizer': "adam",
                                           'learning_rate': 0.00002,
                                           'pretrained_bert_name': "bert-base-multilingual-uncased",
                                           'max_seq_len': 80,
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
                                           'evaluate_begin': 0,
                                           'cross_validate_fold': -1
                                           # split train and test datasets into 5 folds and repeat 3 training
                                           }

_classification_param_dict_chinese = {'model': BERT,
                                      'optimizer': "adam",
                                      'learning_rate': 0.00002,
                                      'pretrained_bert_name': "bert-base-chinese",
                                      'max_seq_len': 80,
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
                                      'evaluate_begin': 0,
                                      'cross_validate_fold': -1
                                      # split train and test datasets into 5 folds and repeat 3 training
                                      }

_classification_param_dict_glove = {'model': LSTM,
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
                                    'cross_validate_fold': -1
                                    }


def get_classification_param_dict_template():
    return copy.deepcopy(_classification_param_dict_template)


def get_classification_param_dict_base():
    return copy.deepcopy(_classification_param_dict_base)


def get_classification_param_dict_english():
    return copy.deepcopy(_classification_param_dict_english)


def get_classification_param_dict_chinese():
    return copy.deepcopy(_classification_param_dict_chinese)


def get_classification_param_dict_multilingual():
    return copy.deepcopy(_classification_param_dict_multilingual)


def get_classification_param_dict_glove():
    return copy.deepcopy(_classification_param_dict_glove)
