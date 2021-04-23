# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from .convert_dataset_for_inferring import convert_dataset_for_inferring
from .functional import train, load_trained_model
from .functional import print_usages
from .batch_inferring.samples import get_samples

#  tunable hyper-parameters:
# model_name = "slide_lcfs_bert", # optional: lcf_bert, lcfs_bert, bert_spc, bert_base
# dataset = "laptop"
# optimizer = "adam"
# learning_rate = 0.00002
# pretrained_bert_name = "bert-base-uncased"
# use_dual_bert = False
# use_bert_spc = True
# max_seq_len = 80
# SRD = 2
# lcf = "cdw"
# window = "lr"
# distance_aware_window = True
# dropout = 0.1
# l2reg = 0.00001
# batch_size = 16

# parameters only for training:
# num_epoch = 3
