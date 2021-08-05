# -*- coding: utf-8 -*-
# file: train_apc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, apc_config_handler, ABSADatasetList, APCModelList

param_dict = apc_config_handler.get_apc_param_dict_chinese()
param_dict['evaluate_begin'] = 3
param_dict['dropout'] = 0.5
param_dict['l2reg'] = 0.0001
param_dict['model'] = APCModelList.FAST_LCF_BERT
save_path = 'state_dict'
chinese_sets = ABSADatasetList.Chinese
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=chinese_sets,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                            auto_device=True  # automatic choose CUDA or CPU
                            )
