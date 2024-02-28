# -*- coding: utf-8 -*-
# file: train.py
# time: 11:30 2023/3/13
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import os
import warnings

from pyabsa.tasks import UniversalSentimentAnalysis as USA

warnings.filterwarnings("ignore")

config = USA.USAConfigManager.get_usa_config_chinese()
config.model = USA.USAModelList.GenerationModel
config.evaluate_begin = 0
config.max_seq_len = 256
config.batch_size = 8
config.pretrained_bert = "google/flan-t5-base"
# config.pretrained_bert = 'allenai/tk-instruct-base-def-pos'
config.log_step = -1
config.l2reg = 1e-8
config.num_epoch = 20
config.seed = 42
config.use_bert_spc = True
config.use_amp = False
config.cache_dataset = False
config.cross_validate_fold = -1

# chinese_sets = USA.USADatasetList.Laptop14
chinese_sets = USA.USADatasetList.Multilingual

usa_model = USA.USATrainer(
    config=config,
    dataset=chinese_sets,
    checkpoint_save_mode=1,
    auto_device=True,
    load_aug=False,
).load_trained_model()

outputs = usa_model.model.evaluate()
print(outputs)
