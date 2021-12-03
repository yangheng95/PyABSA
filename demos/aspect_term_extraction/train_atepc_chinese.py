# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config_chinese = ATEPCConfigManager.get_atepc_config_chinese()
atepc_config_chinese.log_step = 100
atepc_config_chinese.model = ATEPCModelList.FAST_LCF_ATEPC
atepc_config_chinese.evaluate_begin = 5
atepc_config_chinese.l2reg = 1e-6
atepc_config_chinese.num_epoch = 15
atepc_config_chinese.cache_dataset = True

chinese_sets = ABSADatasetList.Chinese

aspect_extractor = Trainer(config=atepc_config_chinese,
                           dataset=chinese_sets,
                           checkpoint_save_mode=1,
                           auto_device=True
                           ).load_trained_model()
