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
from pyabsa import train_apc, apc_config_handler

from pyabsa.model_utils import APCModelList
from pyabsa import ABSADatasets

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.BERT_BASE
apc_param_dict_english['evaluate_begin'] = 10
apc_param_dict_english['similarity_threshold'] = 1
apc_param_dict_english['max_seq_len'] = 80

apc_param_dict_english['dropout'] = 0
apc_param_dict_english['log_step'] = 10
apc_param_dict_english['num_epoch'] = 11
apc_param_dict_english['l2reg'] = 0.0001
apc_param_dict_english['seed'] = {1, 2, 3}
apc_param_dict_english['cross_validate_fold'] = -1  # disable cross_validate

laptop14 = ABSADatasets.Laptop14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant14 = ABSADatasets.Restaurant14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant14,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant15 = ABSADatasets.Restaurant15
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant15,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant16 = ABSADatasets.Restaurant16
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant16,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

apc_param_dict_english['lcf'] = 'cdm'
laptop14 = ABSADatasets.Laptop14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant14 = ABSADatasets.Restaurant14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant14,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant15 = ABSADatasets.Restaurant15
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant15,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
restaurant16 = ABSADatasets.Restaurant16
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=restaurant16,     # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )


#
# # 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
# sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}
# sent_classifier.set_sentiment_map(sentiment_map)
