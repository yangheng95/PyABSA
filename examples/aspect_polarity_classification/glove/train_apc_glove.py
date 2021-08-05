# -*- coding: utf-8 -*-
# project: PyABSA
# file: train_apc_glove.py
# time: 2021/7/18
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc, apc_config_handler, ABSADatasetList
from pyabsa.model_utils import APCModelList

########################################################################################################################
#                To use GloVe-based models, you should put the GloVe embedding into the dataset path                   #
#              or if you can access to Google, it will automatic download GloVe embedding if necessary                 #
########################################################################################################################


# Put glove embedding under current path first if you dont want to download GloVe embedding
save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_glove()
apc_param_dict_english['model'] = APCModelList.GloVeAPCModelList.ASGCN
apc_param_dict_english['max_seq_len'] = 85
apc_param_dict_english['cross_validate_fold'] = -1  # disable cross_validate, enable in {5, 10}

Dataset = ABSADatasetList.Restaurant14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=Dataset,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model if test set is available
                            auto_device=True  # automatic choose CUDA or CPU
                            )
