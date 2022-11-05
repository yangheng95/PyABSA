# -*- coding: utf-8 -*-
# file: trainer.py
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

config = APCConfigManager.get_apc_config_chinese()
config.evaluate_begin = 4
config.dropout = 0
config.l2reg = 1e-5
config.seed = random.randint(0, 10000)
config.model = APCModelList.FAST_LCF_BERT
# configuration_class.spacy_model = 'zh_core_web_sm'
# chinese_sets = ABSADatasetList.Chinese
chinese_sets = ABSADatasetList.Chinese
# chinese_sets = ABSADatasetList.MOOC
sent_classifier = Trainer(config=config,  # set configuration_class=None to use default model
                          dataset=chinese_sets,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,
                          auto_device=True  # automatic choose CUDA or CPU
                          ).load_trained_model()
