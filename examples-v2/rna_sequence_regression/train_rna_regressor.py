# -*- coding: utf-8 -*-
# file: train_rna_classifier.py
# time: 22/10/2022 16:36
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.

import random
from pyabsa import RNARegression as RNAR
from pyabsa.utils.data_utils.dataset_item import DatasetItem

config = RNAR.RNARConfigManager.get_rnar_config_glove()
config.model = RNAR.BERTRNARModelList.BERT_MLP
# config.pretrained_bert = 'rna_decay_bpe_tokenizer'
config.pretrained_bert = "roberta-base"
config.num_epoch = 1
config.evaluate_begin = 0
config.max_seq_len = 128
config.hidden_dim = 768
config.embed_dim = 768
config.cache_dataset = False
# config.cache_dataset = True
config.dropout = 0.5
config.num_lstm_layer = 1
config.seed = [random.randint(0, 10000) for _ in range(3)]
config.log_step = -1
config.l2reg = 0.001
config.do_lower_case = False
config.save_last_ckpt_only = True
config.sigmoid_regression = False

dataset = DatasetItem("decay_rate")
sent_classifier = RNAR.RNARTrainer(
    config=config, dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()
