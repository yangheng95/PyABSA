# -*- coding: utf-8 -*-
# file: train_atepc_multilingual.py
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

config = ATEPCConfigManager.get_atepc_config_multilingual()
config.evaluate_begin = 5
multilingual = ABSADatasetList.Multilingual
aspect_extractor = Trainer(config=config,
                           dataset=multilingual,
                           checkpoint_save_mode=1,
                           auto_device=True
                           )
