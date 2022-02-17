# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import warnings
from pyabsa import GloVeAPCModelList
from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList, BERTBaselineAPCModelList

warnings.filterwarnings('ignore')

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = BERTBaselineAPCModelList.ASGCN_BERT
# apc_config_english = APCConfigManager.get_apc_config_glove()
# apc_config_english.model = GloVeAPCModelList.ASGCN
apc_config_english.num_epoch = 25
apc_config_english.patience = 10
apc_config_english.evaluate_begin = 2
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0
apc_config_english.seed = [1, 2, 3]
apc_config_english.log_step = 10
apc_config_english.l2reg = 1e-8
apc_config_english.learning_rate = 1e-5
apc_config_english.lsa = False
apc_config_english.dynamic_truncate = True
apc_config_english.cache_dataset = False
apc_config_english.srd_alignment = True

# Dataset = '100.CustomDataset'
Dataset = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,
                          checkpoint_save_mode=0,
                          auto_device=True
                          ).load_trained_model()

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = BERTBaselineAPCModelList.ASGCN_BERT
# apc_config_english = APCConfigManager.get_apc_config_glove()
# apc_config_english.model = GloVeAPCModelList.ASGCN
apc_config_english.num_epoch = 25
apc_config_english.patience = 10
apc_config_english.evaluate_begin = 2
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80
apc_config_english.dropout = 0
apc_config_english.seed = [1, 2, 3]
apc_config_english.log_step = 10
apc_config_english.l2reg = 1e-8
apc_config_english.learning_rate = 1e-5
apc_config_english.lsa = True
apc_config_english.dynamic_truncate = True
apc_config_english.cache_dataset = False
apc_config_english.srd_alignment = True

# Dataset = '100.CustomDataset'
Dataset = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,
                          checkpoint_save_mode=0,
                          auto_device=True
                          ).load_trained_model()

