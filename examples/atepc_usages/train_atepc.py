# -*- coding: utf-8 -*-
# file: run_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

#  fine-tuning on custom dataset (w/o test dataset)

from pyabsa import train_atepc

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'lcf_atepc',
              'batch_size': 16,
              'seed': 1,
              'device': 'cuda',
              'num_epoch': 5,
              'optimizer': "adamw",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': False,
              'max_seq_len': 80,
              'log_step': 5,
              'SRD': 3,
              'lcf': "cdw",
              'dropout': 0,
              'l2reg': 0.00001,
              'polarities_dim': 3
              }

# Mind that polarities_dim = 2 for Chinese datasets, and the 'train_atepc' function only evaluates in last two epochs

train_set_path = 'atepc_datasets/restaurant14'
save_path = '../atepc_usages/state_dict/lcf_atepc_cdw_rest14_without_spc'
aspect_extractor = train_atepc(parameter_dict=param_dict,  # set param_dict=None to use default model
                               dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,
                               auto_evaluate=True,  # evaluate model while training if test set is available
                               auto_device=True  # Auto choose CUDA or CPU
                               )

