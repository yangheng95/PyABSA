# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                          This code is for paper:                                                     #
#      "Back to Reality: Leveraging Pattern-driven Modeling to Enable Affordable Sentiment Dependency Learning"        #
#                      but there are some changes in this paper, and it is under submission                            #
########################################################################################################################
import os
import random

import findfile
from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList, BERTBaselineAPCModelList

import warnings
import autocuda
warnings.filterwarnings('ignore')

device = autocuda.auto_cuda()

# seeds = [random.randint(0, 10000) for _ in range(3)]
# 
# apc_config_english = APCConfigManager.get_apc_config_english()
# apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
# apc_config_english.lcf = 'cdw'
# apc_config_english.similarity_threshold = 1
# apc_config_english.max_seq_len = 80
# apc_config_english.dropout = 0.5
# apc_config_english.optimizer = 'adam'
# apc_config_english.cache_dataset = False
# apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
# apc_config_english.hidden_dim = 768
# apc_config_english.embed_dim = 768
# apc_config_english.num_epoch = 30
# apc_config_english.log_step = 5
# apc_config_english.SRD = 3
# apc_config_english.learning_rate = 1e-5
# apc_config_english.batch_size = 16
# apc_config_english.evaluate_begin = 0
# apc_config_english.l2reg = 1e-8
# apc_config_english.seed = seeds
# apc_config_english.MV = MetricVisualizer()
# 
# # for f in findfile.find_cwd_files('.augment.ignore'):
# #     os.rename(f, f.replace('.augment.ignore', '.augment'))
# 
# apc_config_english.cross_validate_fold = -1  # disable cross_validate
# 
# Laptop14 = ABSADatasetList.Laptop14
# Trainer(config=apc_config_english,
#         dataset=Laptop14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant14 = ABSADatasetList.Restaurant14
# Trainer(config=apc_config_english,
#         dataset=Restaurant14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant15 = ABSADatasetList.Restaurant15
# Trainer(config=apc_config_english,
#         dataset=Restaurant15,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant16 = ABSADatasetList.Restaurant16
# Trainer(config=apc_config_english,
#         dataset=Restaurant16,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# apc_config_english.patience = 5
# MAMS = ABSADatasetList.MAMS
# Trainer(config=apc_config_english,
#         dataset=MAMS,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# 
# 
# seeds = [random.randint(0, 10000) for _ in range(3)]
# 
# apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
# apc_config_english = APCConfigManager.get_apc_config_english()
# apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
# apc_config_english.lcf = 'cdw'
# apc_config_english.similarity_threshold = 1
# apc_config_english.max_seq_len = 80
# apc_config_english.dropout = 0
# apc_config_english.optimizer = 'adam'
# apc_config_english.cache_dataset = False
# apc_config_english.pretrained_bert = 'roberta-base'
# apc_config_english.hidden_dim = 768
# apc_config_english.embed_dim = 768
# apc_config_english.num_epoch = 30
# apc_config_english.log_step = 5
# apc_config_english.SRD = 3
# apc_config_english.learning_rate = 2e-5
# apc_config_english.batch_size = 16
# apc_config_english.evaluate_begin = 3
# apc_config_english.l2reg = 1e-5
# apc_config_english.seed = seeds
# 
# # for f in findfile.find_cwd_files('.augment.ignore'):
# #     os.rename(f, f.replace('.augment.ignore', '.augment'))
# 
# apc_config_english.cross_validate_fold = -1  # disable cross_validate
# 
# Laptop14 = ABSADatasetList.Laptop14
# Trainer(config=apc_config_english,
#         dataset=Laptop14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant14 = ABSADatasetList.Restaurant14
# Trainer(config=apc_config_english,
#         dataset=Restaurant14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant15 = ABSADatasetList.Restaurant15
# Trainer(config=apc_config_english,
#         dataset=Restaurant15,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant16 = ABSADatasetList.Restaurant16
# Trainer(config=apc_config_english,
#         dataset=Restaurant16,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# apc_config_english.patience = 5
# MAMS = ABSADatasetList.MAMS
# Trainer(config=apc_config_english,
#         dataset=MAMS,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# 
# seeds = [random.randint(0, 10000) for _ in range(3)]
# 
# apc_config_english = APCConfigManager.get_apc_config_english()
# apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
# apc_config_english.lcf = 'cdw'
# apc_config_english.similarity_threshold = 1
# apc_config_english.max_seq_len = 80
# apc_config_english.dropout = 0
# apc_config_english.optimizer = 'adam'
# apc_config_english.cache_dataset = False
# apc_config_english.pretrained_bert = 'bert-base-uncased'
# apc_config_english.hidden_dim = 768
# apc_config_english.embed_dim = 768
# apc_config_english.num_epoch = 30
# apc_config_english.log_step = 5
# apc_config_english.SRD = 3
# apc_config_english.learning_rate = 1e-5
# apc_config_english.batch_size = 16
# apc_config_english.evaluate_begin = 3
# apc_config_english.l2reg = 1e-8
# apc_config_english.seed = seeds
# 
# # for f in findfile.find_cwd_files('.augment.ignore'):
# #     os.rename(f, f.replace('.augment.ignore', '.augment'))
# 
# apc_config_english.cross_validate_fold = -1  # disable cross_validate
# 
# Laptop14 = ABSADatasetList.Laptop14
# Trainer(config=apc_config_english,
#         dataset=Laptop14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# Restaurant14 = ABSADatasetList.Restaurant14
# Trainer(config=apc_config_english,
#         dataset=Restaurant14,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant15 = ABSADatasetList.Restaurant15
# Trainer(config=apc_config_english,
#         dataset=Restaurant15,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# Restaurant16 = ABSADatasetList.Restaurant16
# Trainer(config=apc_config_english,
#         dataset=Restaurant16,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()
# 
# apc_config_english.patience = 5
# MAMS = ABSADatasetList.MAMS
# Trainer(config=apc_config_english,
#         dataset=MAMS,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# apc_config_english.MV = MetricVisualizer()




seeds = [random.randint(0, 10000) for _ in range(3)]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.optimizer = 'adam'
apc_config_english.cache_dataset = False
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.hidden_dim = 768
apc_config_english.embed_dim = 768
apc_config_english.num_epoch = 30
apc_config_english.log_step = 5
apc_config_english.SRD = 3
apc_config_english.lsa=True
apc_config_english.learning_rate = 1e-5
apc_config_english.batch_size = 16
apc_config_english.evaluate_begin = 0
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds
apc_config_english.MV = MetricVisualizer()

# for f in findfile.find_cwd_files('.augment.ignore'):
#     os.rename(f, f.replace('.augment.ignore', '.augment'))

# apc_config_english.cross_validate_fold = -1  # disable cross_validate
#
Laptop14 = ABSADatasetList.Laptop14
Trainer(config=apc_config_english,
        dataset=Laptop14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant14 = ABSADatasetList.Restaurant14
Trainer(config=apc_config_english,
        dataset=Restaurant14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant15 = ABSADatasetList.Restaurant15
Trainer(config=apc_config_english,
        dataset=Restaurant15,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant16 = ABSADatasetList.Restaurant16
Trainer(config=apc_config_english,
        dataset=Restaurant16,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

apc_config_english.patience = 5
MAMS = ABSADatasetList.MAMS
Trainer(config=apc_config_english,
        dataset=MAMS,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )


seeds = [random.randint(0, 10000) for _ in range(3)]

apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0
apc_config_english.optimizer = 'adam'
apc_config_english.cache_dataset = False
apc_config_english.pretrained_bert = 'roberta-base'
apc_config_english.hidden_dim = 768
apc_config_english.embed_dim = 768
apc_config_english.num_epoch = 30
apc_config_english.log_step = 5
apc_config_english.SRD = 3
apc_config_english.lsa=True
apc_config_english.learning_rate = 2e-5
apc_config_english.batch_size = 16
apc_config_english.evaluate_begin = 3
apc_config_english.l2reg = 1e-5
apc_config_english.seed = seeds

# for f in findfile.find_cwd_files('.augment.ignore'):
#     os.rename(f, f.replace('.augment.ignore', '.augment'))

apc_config_english.cross_validate_fold = -1  # disable cross_validate

Laptop14 = ABSADatasetList.Laptop14
Trainer(config=apc_config_english,
        dataset=Laptop14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant14 = ABSADatasetList.Restaurant14
Trainer(config=apc_config_english,
        dataset=Restaurant14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant15 = ABSADatasetList.Restaurant15
Trainer(config=apc_config_english,
        dataset=Restaurant15,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant16 = ABSADatasetList.Restaurant16
Trainer(config=apc_config_english,
        dataset=Restaurant16,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

apc_config_english.patience = 5
MAMS = ABSADatasetList.MAMS
Trainer(config=apc_config_english,
        dataset=MAMS,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()


seeds = [random.randint(0, 10000) for _ in range(3)]

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = BERTBaselineAPCModelList.TNet_LF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0
apc_config_english.optimizer = 'adam'
apc_config_english.cache_dataset = False
apc_config_english.pretrained_bert = 'bert-base-uncased'
apc_config_english.hidden_dim = 768
apc_config_english.embed_dim = 768
apc_config_english.num_epoch = 30
apc_config_english.log_step = 5
apc_config_english.SRD = 3
apc_config_english.lsa=True
apc_config_english.learning_rate = 1e-5
apc_config_english.batch_size = 16
apc_config_english.evaluate_begin = 3
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds

# for f in findfile.find_cwd_files('.augment.ignore'):
#     os.rename(f, f.replace('.augment.ignore', '.augment'))

apc_config_english.cross_validate_fold = -1  # disable cross_validate

Laptop14 = ABSADatasetList.Laptop14
Trainer(config=apc_config_english,
        dataset=Laptop14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()
Restaurant14 = ABSADatasetList.Restaurant14
Trainer(config=apc_config_english,
        dataset=Restaurant14,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant15 = ABSADatasetList.Restaurant15
Trainer(config=apc_config_english,
        dataset=Restaurant15,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

Restaurant16 = ABSADatasetList.Restaurant16
Trainer(config=apc_config_english,
        dataset=Restaurant16,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()

apc_config_english.patience = 5
MAMS = ABSADatasetList.MAMS
Trainer(config=apc_config_english,
        dataset=MAMS,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
apc_config_english.MV = MetricVisualizer()
