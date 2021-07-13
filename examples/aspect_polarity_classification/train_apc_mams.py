# -*- coding: utf-8 -*-
# file: train_apc_mams.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, apc_config_handler

from pyabsa import ABSADatasets

from pyabsa.model_utils import APCModelList

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.SLIDE_LCF_BERT
apc_param_dict_english['evaluate_begin'] = 2
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['dropout'] = 0.5
apc_param_dict_english['log_step'] = 5
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['dynamic_truncate'] = True
apc_param_dict_english['srd_alignment'] = True

Laptop14 = ABSADatasets.MAMS
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=Laptop14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.SLIDE_LCFS_BERT
apc_param_dict_english['evaluate_begin'] = 2
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['dropout'] = 0.5
apc_param_dict_english['log_step'] = 5
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['dynamic_truncate'] = True
apc_param_dict_english['srd_alignment'] = True

Laptop14 = ABSADatasets.MAMS
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=Laptop14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.FAST_LCF_BERT
apc_param_dict_english['evaluate_begin'] = 2
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['dropout'] = 0.5
apc_param_dict_english['log_step'] = 5
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['dynamic_truncate'] = True
apc_param_dict_english['srd_alignment'] = True

Laptop14 = ABSADatasets.MAMS
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=Laptop14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.FAST_LCFS_BERT
apc_param_dict_english['evaluate_begin'] = 2
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['dropout'] = 0.5
apc_param_dict_english['log_step'] = 5
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['dynamic_truncate'] = True
apc_param_dict_english['srd_alignment'] = True

Laptop14 = ABSADatasets.MAMS
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=Laptop14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

