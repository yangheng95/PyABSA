# -*- coding: utf-8 -*-
# file: train_apc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                           train and evaluate on your own datasets (need train and test datasets)                     #
########################################################################################################################

from pyabsa import train_apc

param_dict = {'model_name': 'lcfs_bert',   # {slide_lcfs_bert, slide_lcf_bert, lcf_bert, lcfs_bert, bert_spc, bert_base}
              'batch_size': 16,
              'seed': {36, 6, 86},         # you can use a set of random seeds to train multiple rounds
              # 'seed': 996,               # or use one seed only
              'num_epoch': 10,
              'optimizer': "adam",         # {adam, adamw}
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,      # modeling the local and global context using different BERTs
              'use_bert_spc': True,        # Enable to enhance APC, do not use this parameter in ATE
              'max_seq_len': 80,
              'log_step': 3,               # Evaluate per steps
              'SRD': 3,                    # Distance threshold to calculate local context
              'eta': -1,                   # Eta is valid in [0,1] slide_lcf_bert/slide_lcfs_bert
              'sigma': 0.3,                # Sigma is valid in LCA-Net, ranging in [0,1]
              'lcf': "cdw",                # {cdm, cdw} valid in lcf-bert models
              'window': "lr",              # {lr, l, r} valid in slide_lcf_bert/slide_lcfs_bert
              'dropout': 0,
              'l2reg': 0.00001,
              # 'polarities_dim': 2,       # deprecated, polarities_dim will be automatic detected
              'dynamic_truncate': True     # Dynamic truncate the text according to the position of aspect term
              }


save_path = 'state_dict'

datasets_path = 'datasets/laptop14'                        # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant14'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant15'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant16'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

param_dict = {'model_name': 'slide_lcfs_bert',   # {slide_lcfs_bert, slide_lcf_bert, lcf_bert, lcfs_bert, bert_spc, bert_base}
              'batch_size': 16,
              'seed': {36, 6, 86},         # you can use a set of random seeds to train multiple rounds
              # 'seed': 996,               # or use one seed only
              'num_epoch': 10,
              'optimizer': "adam",         # {adam, adamw}
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,      # modeling the local and global context using different BERTs
              'use_bert_spc': True,        # Enable to enhance APC, do not use this parameter in ATE
              'max_seq_len': 80,
              'log_step': 3,               # Evaluate per steps
              'SRD': 3,                    # Distance threshold to calculate local context
              'eta': -1,                   # Eta is valid in [0,1] slide_lcf_bert/slide_lcfs_bert
              'sigma': 0.3,                # Sigma is valid in LCA-Net, ranging in [0,1]
              'lcf': "cdm",                # {cdm, cdw} valid in lcf-bert models
              'window': "lr",              # {lr, l, r} valid in slide_lcf_bert/slide_lcfs_bert
              'dropout': 0,
              'l2reg': 0.00001,
              'dynamic_truncate': True     # Dynamic truncate the text according to the position of aspect term
              }

save_path = 'state_dict'
datasets_path = 'datasets/laptop14'                        # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant14'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant15'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

datasets_path = 'datasets/restaurant16'                    # file or dir are accepted
sent_classifier = train_apc(parameter_dict=param_dict,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )
