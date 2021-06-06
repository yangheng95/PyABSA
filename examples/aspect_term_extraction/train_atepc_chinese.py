# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################


from pyabsa import train_atepc

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'lcf_atepc',  # {lcf_atepc, rlcf_atepc}
              'batch_size': 16,
              'seed': {996, 7, 666},
              'device': 'cuda',           # overrides auto_device parameter
              'num_epoch': 6,
              'optimizer': "adamw",       # {adam, adamw}
              'learning_rate': 0.00005,
              'pretrained_bert_name': "bert-base-chinese",
              'use_dual_bert': False,     # modeling the local and global context using different BERTs
              'use_bert_spc': False,      # Enable to enhance APC in lcf_atepc,
                                          # not available for ATE or joint task of APC and ATE
              'max_seq_len': 80,
              'log_step': 5,              # Evaluate per steps
              'SRD': 3,                   # Distance threshold to calculate local context
              'lcf': "cdw",               # {cdw, cdm, fusion}
              'dropout': 0,
              'l2reg': 0.00001,
              'evaluate_begin': 5  # evaluate begin with epoch
              # 'polarities_dim': 2      # Deprecated, polarity_dim will be automatically detected
              }

save_path = 'state_dict'

train_set_path = 'atepc_datasets/Chinese'
aspect_extractor = train_atepc(parameter_dict=param_dict,     # set param_dict=None to use default model
                               dataset_path=train_set_path,   # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,            # evaluate model while training if test set is available
                               auto_device=True               # Auto choose CUDA or CPU
                               )

