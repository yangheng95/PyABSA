# -*- coding: utf-8 -*-
# file: train_apc_multilingual.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, apc_config_handler


save_path = 'state_dict'
apc_param_dict_multilingual = apc_config_handler.get_apc_param_dict_multilingual()
apc_param_dict_multilingual['model_name'] = 'bert_spc'
apc_param_dict_multilingual['evaluate_begin'] = 4

datasets_path = 'datasets/apc_datasets/multilingual'  # file or dir are accepted for 'datasets_path'
sent_classifier = train_apc(parameter_dict=apc_param_dict_multilingual,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
