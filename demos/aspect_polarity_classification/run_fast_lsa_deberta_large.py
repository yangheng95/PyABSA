# -*- coding: utf-8 -*-
# file: run_fast_lsa_deberta.py
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

config1 = APCConfigManager.get_apc_config_english()
config1.model = APCModelList.FAST_LSA_T
config1.lcf = 'cdw'
config1.similarity_threshold = 1
config1.max_seq_len = 80
config1.dropout = 0
config1.optimizer = 'adamw'
config1.cache_dataset = False
config1.patience = 30
config1.pretrained_bert = 'deberta-v3-large'
config1.num_epoch = 30
config1.log_step = -1
config1.SRD = 3
config1.learning_rate = 2e-5
config1.batch_size = 16
config1.evaluate_begin = 3
config1.l2reg = 1e-8
config1.seed = seeds
config1.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant14
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant15
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant16
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config1.patience = 5
dataset = ABSADatasetList.MAMS
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config2 = APCConfigManager.get_apc_config_english()
config2.model = APCModelList.FAST_LSA_S
config2.lcf = 'cdw'
config2.similarity_threshold = 1
config2.max_seq_len = 80
config2.dropout = 0
config2.cache_dataset = False
config2.patience = 30
config2.optimizer = 'adamw'
config2.pretrained_bert = 'deberta-v3-large'
config2.num_epoch = 30
config2.log_step = -1
config2.SRD = 3
config2.learning_rate = 2e-5
config2.batch_size = 16
config2.evaluate_begin = 2
config2.l2reg = 1e-8
config2.seed = seeds
config2.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant14
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant15
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

dataset = ABSADatasetList.Restaurant16
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )

config2.patience = 5
dataset = ABSADatasetList.MAMS
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device  # automatic choose CUDA or CPU
        )
#
# config1 = APCConfigManager.get_apc_config_english()
# config1.model = APCModelList.FAST_LSA_T
# config1.lcf = 'cdw'
# config1.similarity_threshold = 1
# config1.max_seq_len = 80
# config1.dropout = 0
# config1.optimizer = 'adamw'
# config1.cache_dataset = False
# config1.patience = 30
# config1.pretrained_bert = 'roberta-base'
# config1.num_epoch = 30
# config1.log_step = -1
# config1.SRD = 3
# config1.learning_rate = 2e-5
# config1.batch_size = 16
# config1.evaluate_begin = 3
# config1.l2reg = 1e-8
# config1.seed = seeds
# config1.cross_validate_fold = -1  # disable cross_validate
#
# dataset = ABSADatasetList.Laptop14
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant14
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant15
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant16
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config1.patience = 5
# dataset = ABSADatasetList.MAMS
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config2 = APCConfigManager.get_apc_config_english()
# config2.model = APCModelList.FAST_LSA_S
# config2.lcf = 'cdw'
# config2.similarity_threshold = 1
# config2.max_seq_len = 80
# config2.dropout = 0
# config2.cache_dataset = False
# config2.patience = 30
# config2.optimizer = 'adamw'
# config2.pretrained_bert = 'roberta-base'
# config2.num_epoch = 30
# config2.log_step = -1
# config2.SRD = 3
# config2.learning_rate = 2e-5
# config2.batch_size = 16
# config2.evaluate_begin = 2
# config2.l2reg = 1e-8
# config2.seed = seeds
# config2.cross_validate_fold = -1  # disable cross_validate
#
# dataset = ABSADatasetList.Laptop14
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant14
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant15
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config1 = APCConfigManager.get_apc_config_english()
# config1.model = APCModelList.FAST_LSA_T
# config1.lcf = 'cdw'
# config1.similarity_threshold = 1
# config1.max_seq_len = 80
# config1.dropout = 0
# config1.optimizer = 'adamw'
# config1.cache_dataset = False
# config1.patience = 30
# config1.pretrained_bert = 'bert-base-uncased'
# config1.num_epoch = 30
# config1.log_step = -1
# config1.SRD = 3
# config1.learning_rate = 2e-5
# config1.batch_size = 16
# config1.evaluate_begin = 3
# config1.l2reg = 1e-8
# config1.seed = seeds
# config1.cross_validate_fold = -1  # disable cross_validate
#
# dataset = ABSADatasetList.Laptop14
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant14
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant15
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant16
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config1.patience = 5
# dataset = ABSADatasetList.MAMS
# config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config1,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config2 = APCConfigManager.get_apc_config_english()
# config2.model = APCModelList.FAST_LSA_S
# config2.lcf = 'cdw'
# config2.similarity_threshold = 1
# config2.max_seq_len = 80
# config2.dropout = 0
# config2.cache_dataset = False
# config2.patience = 30
# config2.optimizer = 'adamw'
# config2.pretrained_bert = 'bert-base-uncased'
# config2.num_epoch = 30
# config2.log_step = -1
# config2.SRD = 3
# config2.learning_rate = 2e-5
# config2.batch_size = 16
# config2.evaluate_begin = 2
# config2.l2reg = 1e-8
# config2.seed = seeds
# config2.cross_validate_fold = -1  # disable cross_validate
#
# dataset = ABSADatasetList.Laptop14
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant14
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# dataset = ABSADatasetList.Restaurant15
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
# dataset = ABSADatasetList.Restaurant16
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
#
# config2.patience = 5
# dataset = ABSADatasetList.MAMS
# config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
#                               trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
# Trainer(config=config2,
#         dataset=dataset,  # train set and test set will be automatically detected
#         checkpoint_save_mode=0,  # =None to avoid save model
#         auto_device=device  # automatic choose CUDA or CPU
#         )
