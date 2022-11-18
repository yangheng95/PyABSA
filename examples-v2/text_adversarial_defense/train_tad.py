# -*- coding: utf-8 -*-
# file: train_tad.py
# time: 2022/11/18 17:06
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import random
import warnings

import findfile

# Transfer Experiments and Multitask Experiments

from pyabsa import TextAdversarialDefense as TAD, DatasetItem

warnings.filterwarnings('ignore')
seeds = [random.randint(1, 10000) for _ in range(1)]


def get_config():
    config = TAD.TADConfigManager.get_tad_config_english()
    config.model = TAD.BERTTADModelList.TADBERT
    config.num_epoch = 5
    # config.pretrained_bert = 'bert-base-uncased'
    config.patience = 5
    config.evaluate_begin = 0
    config.max_seq_len = 80
    config.log_step = -1
    config.dropout = 0.5
    config.learning_rate = 1e-5
    config.cache_dataset = False
    config.seed = seeds
    config.l2reg = 1e-8
    config.cross_validate_fold = -1
    return config


dataset = DatasetItem('SST2')
text_classifier = TAD.TADTrainer(config=get_config(),
                                 dataset=dataset,
                                 checkpoint_save_mode=1,
                                 auto_device=True
                                 ).load_trained_model()

dataset = DatasetItem('AGNews')
text_classifier = TAD.TADTrainer(config=get_config(),
                                 dataset=dataset,
                                 checkpoint_save_mode=1,
                                 auto_device=True
                                 ).load_trained_model()


dataset = DatasetItem('Amazon')
text_classifier = TAD.TADTrainer(config=get_config(),
                                 dataset=dataset,
                                 checkpoint_save_mode=1,
                                 auto_device=True
                                 ).load_trained_model()
