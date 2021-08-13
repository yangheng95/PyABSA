# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

chinese_sets = ABSADatasetList.Chinese
atepc_config_chinese = ATEPCConfigManager.get_atepc_config_chinese()
atepc_config_chinese.log_step = 50
atepc_config_chinese.model = ATEPCModelList.LCF_ATEPC
atepc_config_chinese.evaluate_begin = 5

aspect_extractor = Trainer(config=atepc_config_chinese,
                           dataset=chinese_sets,
                           checkpoint_save_mode=1,
                           auto_device=True
                           )
