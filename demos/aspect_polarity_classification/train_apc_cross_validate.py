# -*- coding: utf-8 -*-
# file: train_apc_cross_validate.py
# time: 2021/6/25
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.SLIDE_LCF_BERT
apc_config_english.evaluate_begin = 0
apc_config_english.num_epoch = 1
apc_config_english.log_step = 100
apc_config_english.max_seq_len = 80
apc_config_english.cross_validate_fold = 5

laptop14 = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=laptop14,
                          checkpoint_save_mode=True,
                          auto_device=True
                          )
