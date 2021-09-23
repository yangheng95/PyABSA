# -*- coding: utf-8 -*-
# file: training.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset_utils should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################


from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

config = APCConfigManager.get_apc_config_chinese()
config.evaluate_begin = 4
config.dropout = 0.5
config.l2reg = 0.0001
config.model = APCModelList.FAST_LCFS_BERT
config.spacy_model = 'zh_core_web_sm'
# chinese_sets = ABSADatasetList.Chinese
chinese_sets = ABSADatasetList.Shampoo
sent_classifier = Trainer(config=config,  # set config=None to use default model
                          dataset=chinese_sets,  # train set and test set will be automatically detected
                          auto_device=True  # automatic choose CUDA or CPU
                          )
