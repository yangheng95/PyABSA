# -*- coding: utf-8 -*-
# file: run_apc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                           train and evaluate on your own datasets (need train and test datasets)                     #
########################################################################################################################
from pyabsa import train_apc

param_dict = {'model_name': 'slide_lcfs_bert',  # optional: lcf_bert, lcfs_bert, bert_spc, bert_base
              'batch_size': 16,
              'seed': {1, 2, 3},  # you can use a set of random seeds to train multiple rounds
              # 'seed': 996,  # or use one seed only
              'device': 'cuda',
              'num_epoch': 6,
              'optimizer': "adam",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': True,
              'max_seq_len': 80,
              'log_step': 3,  # evaluate per steps
              'SRD': 3,
              'eta': -1,
              'sigma': 0.3,
              'lcf': "cdw",
              'window': "lr",
              'dropout': 0,
              'l2reg': 0.00001,
              }

save_path = 'state_dict'
datasets_path = 'datasets/restaurant14'  # file or dir are OK
# datasets_path = 'sum_train.dat'  # file or dir are OK
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=datasets_path,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

save_path = 'state_dict'
datasets_path = 'datasets/laptop14'  # file or dir are OK
# datasets_path = 'sum_train.dat'  # file or dir are OK
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=datasets_path,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

save_path = 'state_dict'
datasets_path = 'datasets/restaurant15'  # file or dir are OK
# datasets_path = 'sum_train.dat'  # file or dir are OK
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=datasets_path,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

save_path = 'state_dict'
datasets_path = 'datasets/restaurant16'  # file or dir are OK
# datasets_path = 'sum_train.dat'  # file or dir are OK
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=datasets_path,  # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

########################################################################################################################
#                                  fine-tuning on custom dataset (w/o test dataset)                                    #
########################################################################################################################


# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'slide_lcf_bert',  # optional: lcf_bert, lcfs_bert, bert_spc, bert_base
              'batch_size': 16,
              'seed': 1,  # you can use a set of random seeds to train multiple rounds
              'device': 'cuda',
              'num_epoch': 6,
              'optimizer': "adam",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': True,
              'max_seq_len': 80,
              'log_step': 3,
              'SRD': 3,
              'eta': -1,
              'sigma': 0.3,
              'lcf': "cdw",
              'window': "lr",
              'dropout': 0,
              'l2reg': 0.00001,
              }

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'bert_base', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'sum_train.dat'
# train_set_path = 'datasets/restaurant15'

save_path = 'state_dict'
sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

param_dict = {'model_name': 'bert_spc', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'sum_train.dat'
save_path = 'state_dict'

sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

param_dict = {'model_name': 'lcf_bert', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'sum_train.dat'
save_path = 'state_dict'

sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

param_dict = {'model_name': 'lcfs_bert', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'sum_train.dat'
model_path_to_save = save_path

sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

param_dict = {'model_name': 'slide_lcf_bert', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'sum_train.dat'
model_path_to_save = save_path

sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )

param_dict = {'model_name': 'slide_lcfs_bert', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 5}
train_set_path = 'datasets/restaurant15'
train_set_path = 'sum_train.dat'
model_path_to_save = save_path

sent_classifier = train_apc(parameter_dict=param_dict,  # set param_dict=None to use default model
                            dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=False,  # evaluate model while training if test set is available
                            auto_device=True  # Auto choose CUDA or CPU
                            )
