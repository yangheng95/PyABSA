# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import random
import warnings

import findfile

# Transfer Experiments and Multitask Experiments

from pyabsa import (
    TCTrainer,
    TADConfigManager,
    TCDatasetList,
    BERTTADModelList,
    TADTrainer,
)
from pyabsa.functional.dataset.dataset_manager import AdvTCDatasetList, DatasetItem

warnings.filterwarnings("ignore")
seeds = [random.randint(1, 10000) for _ in range(1)]


def get_config():
    config = TADConfigManager.get_tad_config_english()
    config.model = BERTTADModelList.TADBERT
    config.num_epoch = 5
    # config.pretrained_bert = 'bert-base-uncased'
    config.patience = 3
    config.evaluate_begin = 0
    config.max_seq_len = 128
    config.log_step = -1
    config.dropout = 0.0
    config.learning_rate = 2e-5
    config.cache_dataset = False
    config.seed = seeds
    config.l2reg = 1e-5
    config.cross_validate_fold = -1
    return config


dataset = DatasetItem("Amazon")
text_classifier = TADTrainer(
    config=get_config(), dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()
dataset = DatasetItem("AmazonBAE")
text_classifier = TADTrainer(
    config=get_config(), dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()
dataset = DatasetItem("AmazonPWWS")
text_classifier = TADTrainer(
    config=get_config(), dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()
dataset = DatasetItem("AmazonTextFooler")
text_classifier = TADTrainer(
    config=get_config(), dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()
