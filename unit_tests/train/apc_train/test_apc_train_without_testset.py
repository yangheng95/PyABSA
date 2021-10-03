# -*- coding: utf-8 -*-
# file: train_without_testset.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import APCTrainer, ABSADatasetList, APCConfigManager

config = APCConfigManager.get_apc_config_base()
config.evaluate_begin = 0
config.pretrained_bert = 'bert-base-multilingual-uncased'

config.num_epoch = 1
config.log_step = 100

multilingual = ABSADatasetList.Multilingual

sent_classifier = APCTrainer(config=config,
                             dataset=multilingual,
                             checkpoint_save_mode=1,
                             auto_device=True
                             )
