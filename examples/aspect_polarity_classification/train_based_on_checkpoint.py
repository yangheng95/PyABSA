# -*- coding: utf-8 -*-
# file: train_based_on_checkpoint.py
# time: 2021/7/28
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import train_apc, apc_config_handler, ABSADatasetList, APCCheckpointManager
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

checkpoint_path = APCCheckpointManager.get_checkpoint('english')
SemEval = ABSADatasetList.SemEval

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,  # set param_dict=None to use default model
                            dataset_path=SemEval,  # train set and test set will be automatically detected
                            from_checkpoint_path=checkpoint_path,
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                            auto_device=True  # automatic choose CUDA or CPU
                            )
