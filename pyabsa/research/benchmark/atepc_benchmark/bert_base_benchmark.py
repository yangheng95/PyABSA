# -*- coding: utf-8 -*-
# file: bert_base_benchmark.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_atepc, get_atepc_param_dict_english

from pyabsa.absa_dataset import Datasets
from pyabsa.models import ATEPCModels

import copy


def run_lcf_atepc_cdw(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _atepc_param_dict_english = copy.deepcopy(get_atepc_param_dict_english())
        _atepc_param_dict_english['model_name'] = ATEPCModels.BERT_BASE
        _atepc_param_dict_english['lcf'] = 'cdw'
        _atepc_param_dict_english['log_step'] = 10
        _atepc_param_dict_english['evaluate_begin'] = 2
    else:
        _atepc_param_dict_english = param_dict

    train_atepc(dataset_path=Datasets.laptop14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant15,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant16,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.phone,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.car,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.camera,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.notebook,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )


def run_lcf_atepc_cdm(param_dict=None):
    if not param_dict:
        print('No optimal hyper-parameters are set, using default params...')
        _atepc_param_dict_english = copy.deepcopy(get_atepc_param_dict_english())
        _atepc_param_dict_english['model_name'] = ATEPCModels.BERT_BASE
        _atepc_param_dict_english['lcf'] = 'cdm'
        _atepc_param_dict_english['log_step'] = 10
        _atepc_param_dict_english['evaluate_begin'] = 2
    else:
        _atepc_param_dict_english = param_dict

    train_atepc(dataset_path=Datasets.laptop14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant14,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant15,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.restaurant16,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.phone,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.car,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.camera,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )

    train_atepc(dataset_path=Datasets.notebook,
                parameter_dict=_atepc_param_dict_english,  # set param_dict=None will use the apc_param_dict as well
                auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                auto_device=True  # automatic choose CUDA or CPU
                )
