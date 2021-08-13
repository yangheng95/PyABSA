# -*- coding: utf-8 -*-
# file: train_based_on_checkpoint.py
# time: 2021/7/28
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.functional import APCCheckpointManager
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.SLIDE_LCF_BERT
apc_config_english.evaluate_begin = 2
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.l2reg = 0.0001
apc_config_english.dynamic_truncate = True
apc_config_english.srd_alignment = True

checkpoint_path = APCCheckpointManager.get_checkpoint('english')
SemEval = ABSADatasetList.SemEval

sent_classifier = Trainer(config=apc_config_english,
                          dataset=SemEval,
                          from_checkpoint=checkpoint_path,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )
