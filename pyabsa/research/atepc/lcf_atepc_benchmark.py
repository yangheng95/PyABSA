# -*- coding: utf-8 -*-
# file: lcf_atepc_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_atepc, get_apc_param_dict_english

from pyabsa.dataset import *

import copy


def run_lcf_atepc_cdw():
    _atepc_param_dict_english = copy.deepcopy(get_apc_param_dict_english())
    _atepc_param_dict_english['model_name'] = 'lcf_atepc'
    _atepc_param_dict_english['lcf'] = 'cdw'
    _atepc_param_dict_english['evaluate_begin'] = 2

    train_atepc(dataset_path=laptop14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant15,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant16,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=phone,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=car,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=camera,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=notebook,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )


def run_lcf_atepc_cdm():
    _atepc_param_dict_english = copy.deepcopy(get_apc_param_dict_english())
    _atepc_param_dict_english['model_name'] = 'lcf_atepc'
    _atepc_param_dict_english['lcf'] = 'cdm'
    _atepc_param_dict_english['evaluate_begin'] = 2

    train_atepc(dataset_path=laptop14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant15,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=restaurant16,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=phone,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=car,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=camera,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=notebook,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )
