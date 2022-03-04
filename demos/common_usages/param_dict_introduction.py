# -*- coding: utf-8 -*-
# file: param_dict_introduction.py
# time: 2021/6/20
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import APCModelList

# Here are the parameters you can alter
param_dict = {
    # please choose pre-defined model form model list
    'model': APCModelList.LCF_BERT,
    'batch_size': 16,
    'seed': {36, 6, 86},  # you can use a set of random seeds or just one seed to train multiple rounds
    'num_epoch': 10,
    'optimizer': "adam",  # {adam, adamw}
    'learning_rate': 0.00002,
    'pretrained_bert_name': "bert-base-chinese",  # choose a suitable pretrained BERT to train on your dataset_utils
    'use_bert_spc': True,  # Enable to enhance APC, do not use this parameter in ATE or ATEPC
    'max_seq_len': 80,
    'log_step': 10,  # Evaluate per steps
    'SRD': 3,  # Distance threshold to calculate local context
    'eta': -1,  # Eta is valid in [0,1] slide_lcf_bert/slide_lcfs_bert
    'sigma': 0.3,  # Sigma is valid in LCA-Net, ranging in [0,1]
    'lcf': "cdw",  # {cdm, cdw} valid in lcf-bert model
    'window': "lr",  # {lr, l, r} valid in slide_lcf_bert/slide_lcfs_bert
    'dropout': 0,
    'l2reg': 0.00001,
    'dynamic_truncate': True,  # Dynamic truncate the text according to the position of aspect term
    'evaluate_begin': 6  # evaluate begin with epoch
}
