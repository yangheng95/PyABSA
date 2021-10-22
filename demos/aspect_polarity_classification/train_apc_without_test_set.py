# -*- coding: utf-8 -*-
# file: train_apc_without_test_set.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import APCTrainer, ABSADatasetList, APCConfigManager

config = APCConfigManager.get_apc_config_base()

multilingual = ABSADatasetList.Multilingual

sent_classifier = APCTrainer(config=config,
                             dataset=multilingual,
                             checkpoint_save_mode=1,
                             auto_device=True
                             ).load_trained_model()
