# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import warnings

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

warnings.filterwarnings("ignore")

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.FAST_LSA_T_V2
apc_config_english.num_epoch = 30
apc_config_english.evaluate_begin = 2
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.seed = 2672
apc_config_english.log_step = -1
apc_config_english.l2reg = 1e-8
apc_config_english.cache_dataset = False
apc_config_english.dynamic_truncate = True
apc_config_english.srd_alignment = True

Dataset = 'kaggle'
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,
                          checkpoint_save_mode=1,
                          auto_device=True,
                          # load_aug=True
                          ).load_trained_model()

examples = [
    'Strong build though which really adds to its [ASP]durability[ASP] .',  # !sent! Positive
    'Strong [ASP]build[ASP] though which really adds to its durability . !sent! Positive',
    'The [ASP]battery life[ASP] is excellent - 6-7 hours without charging . !sent! Positive',
    'I have had my computer for 2 weeks already and it [ASP]works[ASP] perfectly . !sent! Positive',
    'And I may be the only one but I am really liking [ASP]Windows 8[ASP] . !sent! Positive',
]

inference_sets = examples

for ex in examples:
    result = sent_classifier.infer(ex, print_result=True)
