# -*- coding: utf-8 -*-
# file: train_apc.py
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

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = APCModelList.BERT_SPC
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80

apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.num_epoch = 10
apc_config_english.evaluate_begin = 2
apc_config_english.l2reg = 0.0005
apc_config_english.seed = {1, 2, 3}
apc_config_english.cross_validate_fold = -1  # disable cross_validate
# apc_config_english.use_syntax_based_SRD = True

SemEval = ABSADatasetList.SemEval
sent_classifier = Trainer(config=apc_config_english,  # set config=None will use the apc_config as well
                          dataset=SemEval,  # train set and test set will be automatically detected
                          save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                          auto_device=True  # automatic choose CUDA or CPU
                          )
