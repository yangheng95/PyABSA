# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList, APCTrainer

from pyabsa import AspectPolarityClassification as APC
from pyabsa.augmentation import auto_aspect_sentiment_classification_augmentation
import warnings

warnings.filterwarnings("ignore")

for dataset in [
    # 'SNLI',
    "MNLI"
]:
    for model in [APC.APCModelList.BERT_MLP]:
        config = APC.APCConfigManager.get_apc_config_english()
        config.model = model
        # config.pretrained_bert = "microsoft/deberta-v3-base"
        config.pretrained_bert = "bert-base-uncased"
        config.evaluate_begin = 0
        config.max_seq_len = 80
        config.num_epoch = 10
        config.patience = 3
        config.log_step = 100
        config.dropout = 0
        config.cache_dataset = False
        config.l2reg = 1e-8
        config.lsa = False
        config.use_amp = True
        config.verbose = False
        config.seed = [random.randint(0, 10000) for _ in range(3)]

        APCTrainer(
            config=config, dataset=dataset, auto_device=True
        ).load_trained_model()
