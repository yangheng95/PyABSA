# -*- coding: utf-8 -*-
# file: bert_spc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc, apc_param_dict_english

from pyabsa.dataset import *

import copy


def run_bert_spc_cdw():
    _apc_param_dict_english = copy.deepcopy(apc_param_dict_english)
    _apc_param_dict_english['model_name'] = 'bert_spc'
    _apc_param_dict_english['lcf'] = 'cdw'
    _apc_param_dict_english['evaluate_begin'] = 2

    train_apc(dataset_path=laptop14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant15,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant16,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )


def run_bert_spc_cdm():
    _apc_param_dict_english = copy.deepcopy(apc_param_dict_english)
    _apc_param_dict_english['model_name'] = 'bert_spc'
    _apc_param_dict_english['lcf'] = 'cdm'
    _apc_param_dict_english['evaluate_begin'] = 2

    train_apc(dataset_path=laptop14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant14,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant15,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )

    train_apc(dataset_path=restaurant16,
              parameter_dict=_apc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
              auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
              auto_device=True  # automatic choose CUDA or CPU
              )
