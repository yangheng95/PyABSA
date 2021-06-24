# -*- coding: utf-8 -*-
# file: lcfs_bert_large_benchmark.py
# time: 2021/6/24
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc, apc_config_handler

from pyabsa import ABSADatasets
from pyabsa.model_utils import APCModelList

import copy


def run_lcfs_bert_large_cdw(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _apc_param_dict_english = copy.deepcopy(apc_config_handler.get_apc_param_dict_english())
        _apc_param_dict_english['model_name'] = APCModelList.LCFS_BERT_LARGE
        _apc_param_dict_english['evaluate_begin'] = 1
        _apc_param_dict_english['similarity_threshold'] = 1
        _apc_param_dict_english['dropout'] = 0.5
        _apc_param_dict_english['log_step'] = 5
        _apc_param_dict_english['num_epoch'] = 10
        _apc_param_dict_english['l2reg'] = 0.0001
    else:
        _apc_param_dict_english = param_dict

    train_apc(dataset_path=ABSADatasets.Laptop14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant15,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant16,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )


def run_lcfs_bert_large_cdm(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _apc_param_dict_english = copy.deepcopy(apc_config_handler.get_apc_param_dict_english())
        _apc_param_dict_english['model_name'] = APCModelList.LCFS_BERT_LARGE
        _apc_param_dict_english['evaluate_begin'] = 1
        _apc_param_dict_english['similarity_threshold'] = 1
        _apc_param_dict_english['dropout'] = 0.5
        _apc_param_dict_english['log_step'] = 5
        _apc_param_dict_english['num_epoch'] = 10
        _apc_param_dict_english['l2reg'] = 0.0001
        _apc_param_dict_english['lcf'] = 'cdm'
    else:
        _apc_param_dict_english = param_dict

    train_apc(dataset_path=ABSADatasets.Laptop14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant15,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=ABSADatasets.Restaurant16,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )
