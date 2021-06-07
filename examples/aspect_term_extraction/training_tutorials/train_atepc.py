# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################


from pyabsa import train_atepc, atepc_param_dict_english


save_path = '../state_dict'

# Mind that the 'train_atepc' function only evaluates in last few epochs
train_set_path = '../atepc_datasets/SemEval/restaurant15'
aspect_extractor = train_atepc(parameter_dict=atepc_param_dict_english,      # set param_dict=None to use default model
                               dataset_path=train_set_path,    # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,             # evaluate model while training_tutorials if test set is available
                               auto_device=True                # Auto choose CUDA or CPU
                               )

# you can construct param_dict by your welling
# param_dict = {'model_name': 'lcf_atepc',
#               'batch_size': 16,
#               'seed': {996},
#               'num_epoch': 6,
#               'optimizer': "adam",    # {adam, adamw}
#               'learning_rate': 0.00003,
#               'pretrained_bert_name': "bert-base-uncased",
#               'use_dual_bert': False,  # modeling the local and global context using different BERTs
#               'use_bert_spc': False,   # enable to enhance APC, not available for ATE or joint task of APC and ATE
#               'max_seq_len': 80,
#               'log_step': 5,           # evaluate per steps
#               'SRD': 3,                # distance threshold to calculate local context
#               'use_syntax_based_SRD': True,   # force to use syntax-based semantic-relative distance in all lcf-based models
#               'lcf': "cdw",            # {cdw, cdm, fusion}
#               'dropout': 0,
#               'l2reg': 0.00001,
#               'evaluate_begin': 4  # evaluate begin with epoch
#               # 'polarities_dim': 3      # deprecated, polarity_dim will be automatically detected
#               }
# train_set_path = '../atepc_datasets/SemEval/laptop14'
# aspect_extractor = train_atepc(parameter_dict=param_dict,      # set param_dict=None to use default model
#                                dataset_path=train_set_path,    # file or dir, dataset(s) will be automatically detected
#                                model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
#                                auto_evaluate=True,             # evaluate model while training_tutorials if test set is available
#                                auto_device=True                # Auto choose CUDA or CPU
#                                )


