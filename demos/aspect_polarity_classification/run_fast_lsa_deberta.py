# -*- coding: utf-8 -*-
# file: exp0.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                          This code is for paper:                                                     #
#      "Back to Reality: Leveraging Pattern-driven Modeling to Enable Affordable Sentiment Dependency Learning"        #
#                      but there are some changes in this paper, and it is under submission                            #
########################################################################################################################
import random

import autocuda
from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

import warnings

warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(3)]
device = autocuda.auto_cuda()

config = APCConfigManager.get_apc_config_english()
config.model = APCModelList.FAST_LSA_T_V2
config.lcf = 'cdw'
config.similarity_threshold = 1
config.max_seq_len = 80
config.dropout = 0.5
config.optimizer = 'adamw'
config.cache_dataset = False
config.patience = 30
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.num_epoch = 30
config.log_step = 5
config.SRD = 3
config.lsa = True
config.eta = 0.5
config.learning_rate = 2e-5
config.batch_size = 16
config.evaluate_begin = 0
config.l2reg = 1e-5
config.seed = seeds
config.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config.log_step = -1
config.patience = 5
dataset = ABSADatasetList.MAMS
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config = APCConfigManager.get_apc_config_english()
config.model = APCModelList.FAST_LSA_S_V2
config.lcf = 'cdw'
config.similarity_threshold = 1
config.max_seq_len = 80
config.dropout = 0.5
config.cache_dataset = False
config.patience = 30
config.optimizer = 'adamw'
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.num_epoch = 30
config.log_step = 5
config.SRD = 3
config.lsa = True
config.eta = 0.5
config.learning_rate = 2e-5
config.batch_size = 16
config.evaluate_begin = 2
config.l2reg = 1e-5
config.seed = seeds
config.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config.log_step = -1
config.patience = 5
dataset = ABSADatasetList.MAMS
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config = APCConfigManager.get_apc_config_english()
config.model = APCModelList.BERT_SPC_V2
config.lcf = 'cdw'
config.similarity_threshold = 1
config.max_seq_len = 80
config.dropout = 0.5
config.cache_dataset = False
config.patience = 30
config.optimizer = 'adamw'
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.num_epoch = 30
config.log_step = 5
config.SRD = 3
config.lsa = True
config.eta = 0.5
config.learning_rate = 2e-5
config.batch_size = 16
config.evaluate_begin = 2
config.l2reg = 1e-5
config.seed = seeds
config.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant14
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config.log_step = -1
config.patience = 5
dataset = ABSADatasetList.MAMS
config.MV = MetricVisualizer(config.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                             trial_tag_list=[config.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
