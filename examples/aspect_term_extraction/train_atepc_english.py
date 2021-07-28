# -*- coding: utf-8 -*-
# file: train_atepc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################


from pyabsa import train_atepc, atepc_config_handler

from pyabsa import ABSADatasetList

from pyabsa import ATEPCModelList

save_path = 'state_dict'

param_dict = atepc_config_handler.get_atepc_param_dict_english()
param_dict['model'] = ATEPCModelList.LCF_ATEPC
param_dict['evaluate_begin'] = 5
param_dict['num_epoch'] = 6
param_dict['log_step'] = 100
semeval = ABSADatasetList.SemEval
aspect_extractor = train_atepc(parameter_dict=param_dict,      # set param_dict=None to use default model
                               dataset_path=semeval,           # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,             # evaluate model while training_tutorials if test set is available
                               auto_device=True                # Auto choose CUDA or CPU
                               )

