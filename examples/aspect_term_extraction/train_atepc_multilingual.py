# -*- coding: utf-8 -*-
# file: train_atepc_multilingual.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################


from pyabsa import train_atepc, atepc_config_handler, ABSADatasetList

param_dict = atepc_config_handler.get_atepc_param_dict_multilingual()
param_dict['evaluate_begin'] = 5
save_path = 'state_dict'
multilingual = ABSADatasetList.Multilingual
aspect_extractor = train_atepc(parameter_dict=param_dict,  # set param_dict=None to use default model
                               dataset_path=multilingual,  # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                               auto_device=True  # Auto choose CUDA or CPU
                               )
