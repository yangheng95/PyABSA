# -*- coding: utf-8 -*-
# file: train_apc_multilingual.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################


from pyabsa.framework import Trainer
from pyabsa.framework import APCConfigManager
from pyabsa.framework import ABSADatasetList
from pyabsa.framework import APCModelList

apc_config_multilingual = APCConfigManager.get_apc_config_multilingual()
apc_config_multilingual.model = APCModelList.FAST_LSA_T_V2
# apc_config_multilingual.pretrained_bert = 'bert-base-multilingual-cased'
apc_config_multilingual.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
apc_config_multilingual.log_step = -1
apc_config_multilingual.l2reg = 1e-5
apc_config_multilingual.optimizer = 'adam'
apc_config_multilingual.cache_dataset = False
apc_config_multilingual.seed = random.randint(0, 10000)


datasets_path = ABSADatasetList.Multilingual
sent_classifier = Trainer(config=apc_config_multilingual,
                          dataset=datasets_path,
                          checkpoint_save_mode=1,  # save state_dict instead of model
                          auto_device=True,  # auto-select cuda device
                          # load_aug=True,  # trainer using augment data
                          ).load_trained_model()
