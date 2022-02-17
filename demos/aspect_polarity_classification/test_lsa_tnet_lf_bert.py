# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import random
import warnings

import autocuda
from metric_visualizer import MetricVisualizer

from pyabsa import GloVeAPCModelList
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList, BERTBaselineAPCModelList

warnings.filterwarnings('ignore')

device = autocuda.auto_cuda()

seeds = [random.randint(0, 10000) for _ in range(5)]

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
apc_config_english.log_step = -1
apc_config_english.SRD = 3
apc_config_english.learning_rate = 1e-5
apc_config_english.batch_size = 16
apc_config_english.num_epoch = 25
apc_config_english.evaluate_begin = 5
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds
apc_config_english.cross_validate_fold = -1  # disable cross_validate


MV = MetricVisualizer(name='test', trial_tag='Model', trial_tag_list=['TNet w/o LSA', 'TNet w/ LSA'])
apc_config_english.MV = MV

for dataset in [
        ABSADatasetList.Laptop14,
        ABSADatasetList.Restaurant14,
        ABSADatasetList.Restaurant15,
        ABSADatasetList.Restaurant16,
        # ABSADatasetList.MAMS
    ]:
        sent_classifier = Trainer(config=apc_config_english,
                                  dataset=dataset,
                                  checkpoint_save_mode=0,
                                  auto_device=True
                                  ).load_trained_model()
        apc_config_english.MV.next_trial()
apc_config_english.MV.summary()
apc_config_english.MV.violin_plot_by_trial()
apc_config_english.MV.box_plot_by_trial()



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
apc_config_english.log_step = -1
apc_config_english.SRD = 3
apc_config_english.learning_rate = 1e-5
apc_config_english.batch_size = 16
apc_config_english.num_epoch = 25
apc_config_english.evaluate_begin = 5
apc_config_english.l2reg = 1e-8
apc_config_english.seed = seeds
apc_config_english.cross_validate_fold = -1  # disable cross_validate


MV = MetricVisualizer(name='test', trial_tag='Model', trial_tag_list=['TNet w/o LSA', 'TNet w/ LSA'])
apc_config_english.MV = MV

for dataset in [
        ABSADatasetList.Laptop14,
        ABSADatasetList.Restaurant14,
        ABSADatasetList.Restaurant15,
        ABSADatasetList.Restaurant16,
        # ABSADatasetList.MAMS
    ]:
        sent_classifier = Trainer(config=apc_config_english,
                                  dataset=dataset,
                                  checkpoint_save_mode=0,
                                  auto_device=True
                                  ).load_trained_model()
        apc_config_english.MV.next_trial()
apc_config_english.MV.summary()
apc_config_english.MV.violin_plot_by_trial()
apc_config_english.MV.box_plot_by_trial()
