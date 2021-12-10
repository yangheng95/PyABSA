# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################


from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.DLCFS_DCA_BERT
apc_config_english.num_epoch = 30
apc_config_english.evaluate_begin = 5
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 120
apc_config_english.dropout = 0.5
apc_config_english.seed = 2672
apc_config_english.log_step = 10
apc_config_english.l2reg = 1e-8
apc_config_english.dynamic_truncate = True
apc_config_english.srd_alignment = True

Dataset = ABSADatasetList.Yelp
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,
                          checkpoint_save_mode=1,
                          auto_device='allcuda'
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
