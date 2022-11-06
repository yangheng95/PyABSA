# -*- coding: utf-8 -*-
# file: exp0.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
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
config1.model = APCModelList.FAST_LSA_T_V2
config1.lcf = 'cdw'
config1.similarity_threshold = 1
config1.max_seq_len = 80
config1.dropout = 0.
config1.cache_dataset = False
config1.patience = 20
config1.optimizer = 'adamw'
config1.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
config1.num_epoch = 50
config1.log_step = 5
config1.SRD = 3
config1.eta = 1
config1.eta_lr = 0.001
config1.lsa = True
config1.learning_rate = 2e-5
config1.batch_size = 16
config1.evaluate_begin = 0
config1.l2reg = 1e-4
config1.seed = seeds
config1.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config1.MV.avg_bar_plot()
config1.MV.box_plot()
# see the usage of metric-visualizer at https://github.com/yangheng95/PyABSA/blob/release/readme/metric_readme.md

dataset = ABSADatasetList.Restaurant14
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config1.MV.avg_bar_plot()
config1.MV.box_plot()

config1.log_step = -1
config1.patience = 15
dataset = ABSADatasetList.MAMS
config1.MV = MetricVisualizer(config1.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config1.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config1,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config1.MV.avg_bar_plot()
config1.MV.box_plot()

config2 = APCConfigManager.get_apc_config_english()
config2.model = APCModelList.FAST_LSA_S_V2
config2.lcf = 'cdw'
config2.similarity_threshold = 1
config2.max_seq_len = 80
config2.dropout = 0.
config2.cache_dataset = False
config2.patience = 20
config2.optimizer = 'adamw'
config2.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
config2.num_epoch = 50
config2.log_step = 5
config2.SRD = 3
config2.eta = 1
config2.eta_lr = 0.001
config2.lsa = True
config2.learning_rate = 2e-5
config2.batch_size = 16
config2.evaluate_begin = 0
config2.l2reg = 1e-4
config2.seed = seeds
config2.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config2.MV.avg_bar_plot()
config2.MV.box_plot()

dataset = ABSADatasetList.Restaurant14
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config2.MV.avg_bar_plot()
config2.MV.box_plot()

config2.log_step = -1
config2.patience = 15
dataset = ABSADatasetList.MAMS
config2.MV = MetricVisualizer(config2.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config2.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config2,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config2.MV.avg_bar_plot()
config2.MV.box_plot()

config3 = APCConfigManager.get_apc_config_english()
config3.model = APCModelList.BERT_SPC_V2
config3.lcf = 'cdw'
config3.similarity_threshold = 1
config3.max_seq_len = 80
config3.dropout = 0.
config3.cache_dataset = False
config3.patience = 20
config3.optimizer = 'adamw'
config3.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
config3.num_epoch = 50
config3.log_step = 5
config3.SRD = 3
config3.eta = 1
config3.eta_lr = 0.001
config3.lsa = True
config3.learning_rate = 2e-5
config3.batch_size = 16
config3.evaluate_begin = 0
config3.l2reg = 1e-4
config3.seed = seeds
config3.cross_validate_fold = -1  # disable cross_validate

dataset = ABSADatasetList.Laptop14
config3.MV = MetricVisualizer(config3.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config3.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config3,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config3.MV.avg_bar_plot()
config3.MV.box_plot()

dataset = ABSADatasetList.Restaurant14
config3.MV = MetricVisualizer(config3.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config3.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config3,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config3.MV.avg_bar_plot()
config3.MV.box_plot()

config3.log_step = -1
config3.patience = 15
dataset = ABSADatasetList.MAMS
config3.MV = MetricVisualizer(config3.model.__name__ + '-' + dataset.dataset_name, trial_tag='Model & Dataset',
                              trial_tag_list=[config3.model.__name__ + '-' + dataset.dataset_name])
Trainer(config=config3,
        dataset=dataset,  # train set and test set will be automatically detected
        checkpoint_save_mode=0,  # =None to avoid save model
        auto_device=device,  # automatic choose CUDA or CPU
        # load_aug=True,  # load augmented data
        )
config3.MV.avg_bar_plot()
config3.MV.box_plot()
