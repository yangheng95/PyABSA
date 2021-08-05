# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################


from pyabsa import train_atepc, atepc_config_handler, ABSADatasetList
from pyabsa.model_utils import ATEPCModelList

save_path = 'state_dict'

chinese_sets = ABSADatasetList.Chinese
atepc_param_dict_chinese = atepc_config_handler.get_atepc_param_dict_chinese()
atepc_param_dict_chinese['log_step'] = 20
atepc_param_dict_chinese['model'] = ATEPCModelList.LCF_ATEPC
atepc_param_dict_chinese['evaluate_begin'] = 5

aspect_extractor = train_atepc(parameter_dict=atepc_param_dict_chinese,  # set param_dict=None to use default model
                               dataset_path=chinese_sets,  # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                               auto_device=True  # Auto choose CUDA or CPU
                               )
