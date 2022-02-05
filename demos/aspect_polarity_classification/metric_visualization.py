# -*- coding: utf-8 -*-
# file: metric_visualization.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import random
from distutils.version import StrictVersion

import autocuda
import numpy as np
import pyabsa
from metric_visualizer import MetricVisualizer

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

import warnings

if not StrictVersion(pyabsa.__version__) > StrictVersion('1.8.15'):
    raise KeyError('This demo can only run on PyABSA > 1.8.15')

warnings.filterwarnings('ignore')

seeds = [random.randint(0, 10000) for _ in range(5)]

eta_candidates = list(np.arange(0, 1.1, 0.1))

device = autocuda.auto_cuda()

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LSA_T
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
    dataset = ABSADatasetList.Laptop14
    Trainer(config=apc_config_english,
            dataset=dataset,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=device  # automatic choose CUDA or CPU
            )
    apc_config_english.MV.next_trail()

apc_config_english.MV.summary(save_path=None, xticks=eta_candidates)
apc_config_english.MV.traj_plot(save_path=None, xticks=eta_candidates)
apc_config_english.MV.violin_plot(save_path=None, xticks=eta_candidates)
apc_config_english.MV.box_plot(save_path=None, xticks=eta_candidates)

save_path = '{}_{}'.format(apc_config_english.model_name, apc_config_english.dataset_name)
apc_config_english.MV.summary(save_path=save_path)
apc_config_english.MV.traj_plot(save_path=save_path, xticks=eta_candidates, xlabel=r'$\eta$')
apc_config_english.MV.violin_plot(save_path=save_path, xticks=eta_candidates, xlabel=r'$\eta$')
apc_config_english.MV.box_plot(save_path=save_path, xticks=eta_candidates, xlabel=r'$\eta$')

