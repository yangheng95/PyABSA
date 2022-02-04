# -*- coding: utf-8 -*-
# file: metric_visualization.py
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

import numpy as np
from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

import warnings

warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(2)]

eta_candidates = np.arange(0, 1.1, 0.1)

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LCF_BERT
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0
apc_config_english.optimizer = 'adam'
apc_config_english.cache_dataset = False
apc_config_english.patience = 10
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.hidden_dim = 768
apc_config_english.embed_dim = 768
apc_config_english.log_step = 10
apc_config_english.SRD = 3
apc_config_english.learning_rate = 1e-5
apc_config_english.batch_size = 16
apc_config_english.num_epoch = 25
apc_config_english.evaluate_begin = 2
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds

apc_config_english.cross_validate_fold = -1  # disable cross_validate

MV = MetricVisualizer()
apc_config_english.MV = MV

for eta in eta_candidates:
    apc_config_english.eta = eta
    Laptop14 = ABSADatasetList.Laptop14
    Trainer(config=apc_config_english,
            dataset=Laptop14,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=True  # automatic choose CUDA or CPU
            )
    apc_config_english.MV.next_trail()

save_path = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)
MV.summary(save_path=None)
MV.traj_plot(save_path=None)
MV.violin_plot(save_path=None)
MV.box_plot(save_path=None)

try:
    MV.summary(save_path=save_path)
    MV.traj_plot(save_path=save_path)
    MV.violin_plot(save_path=save_path)
    MV.box_plot(save_path=save_path)
except Exception as e:
    pass
