# -*- coding: utf-8 -*-
# file: train.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################


from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_multilingual = APCConfigManager.get_apc_config_multilingual()
apc_config_multilingual.model = APCModelList.SLIDE_LCFS_BERT
apc_config_multilingual.pretrained_bert = 'bert-base-multilingual-uncased'
apc_config_multilingual.evaluate_begin = 0
apc_config_multilingual.num_epoch = 1
apc_config_multilingual.log_step = 100

datasets_path = ABSADatasetList.Multilingual
sent_classifier = Trainer(config=apc_config_multilingual,
                          dataset=datasets_path,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )
