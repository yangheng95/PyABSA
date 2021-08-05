# -*- coding: utf-8 -*-
# file: train_atepc_based_on_checkpoint.py
# time: 2021/7/27
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_atepc, atepc_config_handler, ATEPCCheckpointManager, ABSADatasetList, ATEPCModelList

param_dict = atepc_config_handler.get_atepc_param_dict_english()
checkpoint_path = ATEPCCheckpointManager.get_checkpoint(checkpoint_name='english')

param_dict['model'] = ATEPCModelList.LCFS_ATEPC
param_dict['evaluate_begin'] = 0
save_path = 'state_dict'
SemEval = ABSADatasetList.Laptop14
aspect_extractor = train_atepc(parameter_dict=param_dict,  # set param_dict=None to use default model
                               dataset_path=SemEval,  # file or dir, dataset(s) will be automatically detected
                               from_checkpoint_path=checkpoint_path,
                               model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                               auto_device=True  # Auto choose CUDA or CPU
                               )
