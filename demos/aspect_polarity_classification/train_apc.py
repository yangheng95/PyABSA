# -*- coding: utf-8 -*-
# file: train_apc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = [APCModelList.SLIDE_LCF_BERT, APCModelList.FAST_LCF_BERT]
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80

apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
# apc_config_english.pretrained_bert = 'bert-base-multilingual-uncased'
apc_config_english.pretrained_bert = 'bert-base-uncased'
apc_config_english.num_epoch = 10
apc_config_english.batch_size = 16
apc_config_english.evaluate_begin = 0
apc_config_english.l2reg = 0.0005
apc_config_english.seed = {1, 2, 3}
apc_config_english.cross_validate_fold = -1  # disable cross_validate
# apc_config_english.use_syntax_based_SRD = True

Multilingual = ABSADatasetList.Laptop14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Multilingual,  # train set and test set will be automatically detected
                          checkpoint_save_mode=0,  # =None to avoid save model
                          # auto_device=True  # automatic choose CUDA or CPU
                          auto_device='allcuda'  # automatic choose CUDA or CPU
                          # auto_device='cuda:2'  # automatic choose CUDA or CPU
                          ).load_trained_model()
