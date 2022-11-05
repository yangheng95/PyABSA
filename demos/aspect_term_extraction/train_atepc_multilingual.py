# -*- coding: utf-8 -*-
# file: train_atepc_multilingual.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC trainer script                                                  #
########################################################################################################################

from pyabsa.framework import ATEPCConfigManager, Trainer, ABSADatasetList

config = ATEPCConfigManager.get_atepc_config_multilingual()
multilingual = ABSADatasetList.Multilingual
aspect_extractor = Trainer(config=config,
                           dataset=multilingual,
                           ).load_trained_model()
