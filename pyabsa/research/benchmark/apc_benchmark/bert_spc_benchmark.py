# -*- coding: utf-8 -*-
# file: bert_spc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc, apc_config_handler

from pyabsa import ABSADatasets
from pyabsa.models import APCModelList

import copy


def run_bert_spc_cdw(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _apc_param_dict_english = copy.deepcopy(apc_config_handler.get_apc_param_dict_english())
        _apc_param_dict_english['model_name'] = APCModelList.BERT_SPC
        _apc_param_dict_english['lcf'] = 'cdw'
        _apc_param_dict_english['evaluate_begin'] = 1
        _apc_param_dict_english['l2reg'] = 1e-5
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


def run_bert_spc_cdm(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _apc_param_dict_english = copy.deepcopy(apc_config_handler.get_apc_param_dict_english())
        _apc_param_dict_english['model_name'] = APCModelList.BERT_SPC
        _apc_param_dict_english['lcf'] = 'cdm'
        _apc_param_dict_english['evaluate_begin'] = 1
        _apc_param_dict_english['l2reg'] = 1e-5
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
