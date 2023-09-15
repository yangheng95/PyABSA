# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################
import random

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config_chinese = ATEPCConfigManager.get_atepc_config_chinese()
atepc_config_chinese.model = ATEPCModelList.FAST_LCF_ATEPC
atepc_config_chinese.evaluate_begin = 0
atepc_config_chinese.pretrained_bert = "bert-base-chinese"
# atepc_config_chinese.pretrained_bert = 'microsoft/mdeberta-v3-base'
atepc_config_chinese.log_step = -1
atepc_config_chinese.l2reg = 1e-5
atepc_config_chinese.num_epoch = 30
atepc_config_chinese.seed = random.randint(1, 100)
atepc_config_chinese.use_bert_spc = True
atepc_config_chinese.cache_dataset = False

chinese_sets = ABSADatasetList.Chinese

aspect_extractor = Trainer(
    config=atepc_config_chinese,
    dataset=chinese_sets,
    checkpoint_save_mode=1,
    auto_device=True,
).load_trained_model()
