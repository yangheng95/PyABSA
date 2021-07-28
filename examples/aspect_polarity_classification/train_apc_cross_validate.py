# -*- coding: utf-8 -*-
# file: train_apc_cross_validate.py
# time: 2021/6/25
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import train_apc, apc_config_handler

from pyabsa.model_utils import APCModelList
from pyabsa import ABSADatasetList

save_path = 'state_dict'
apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['model'] = APCModelList.SLIDE_LCF_BERT
apc_param_dict_english['evaluate_begin'] = 2
apc_param_dict_english['num_epoch'] = 6
apc_param_dict_english['max_seq_len'] = 80
apc_param_dict_english['cross_validate_fold'] = -1

laptop14 = ABSADatasetList.Laptop14
sent_classifier = train_apc(parameter_dict=apc_param_dict_english,
                            # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,         # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
