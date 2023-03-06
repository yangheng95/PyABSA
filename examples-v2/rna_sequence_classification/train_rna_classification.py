# -*- coding: utf-8 -*-
# file: train_rna_classifier.py
# time: 22/10/2022 16:36
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa import RNAClassification as RNAC
from pyabsa.utils.data_utils.dataset_item import DatasetItem

config = RNAC.RNACConfigManager.get_rnac_config_english()
config.model = RNAC.BERTRNACModelList.BERT_MLP
config.pretrained_bert = "roberta-base"
config.num_epoch = 1
config.evaluate_begin = 0
config.max_seq_len = 10
# config.warmup_step = 1000
config.learning_rate = 1e-5
config.l2reg = 0
config.use_amp = True
config.cache_dataset = False
config.dropout = 0
config.show_metric = True
config.save_last_ckpt_only = True
config.seed = [random.randint(1, 10000) for _ in range(5)]
config.batch_size = 16
# config.batch_size = 32
config.log_step = -1
# config.log_step = 10


dataset = DatasetItem("sfe")

config.sigmoid_regression = False
# dataset = DatasetItem('sfe')
# config.sigmoid_regression = True

sent_classifier = RNAC.RNACTrainer(
    config=config,
    dataset=dataset,
    # from_checkpoint='bert_decay_rate_r2_0.5378',
    checkpoint_save_mode=1,
    auto_device=True,
).load_trained_model()
