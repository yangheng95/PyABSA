# -*- coding: utf-8 -*-
# file: train_apc_ensemble.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_english = APCConfigManager.get_apc_config_english()
apc_config_english.model = [APCModelList.FAST_LSA_S, APCModelList.FAST_LSA_T]
# apc_config_english.model = [APCModelList.FAST_LCF_BERT]
apc_config_english.lcf = 'cdw'
apc_config_english.similarity_threshold = 1
apc_config_english.max_seq_len = 80

apc_config_english.dropout = 0.5
apc_config_english.log_step = 50
apc_config_english.pretrained_bert = 'microsoft/deberta-v3-base'
apc_config_english.num_epoch = 15
apc_config_english.batch_size = 16
apc_config_english.evaluate_begin = 2
apc_config_english.l2reg = 1e-8
apc_config_english.seed = {1, 2, 3}
apc_config_english.cross_validate_fold = -1  # disable cross_validate
# apc_config_english.use_syntax_based_SRD = True

Dataset = ABSADatasetList.Restaurant14
sent_classifier = Trainer(config=apc_config_english,
                          dataset=Dataset,  # train set and test set will be automatically detected
                          checkpoint_save_mode=1,  # =None to avoid save model
                          # auto_device=True  # automatic choose CUDA or CPU
                          auto_device='allcuda'  # automatic choose CUDA or CPU
                          # auto_device='cuda:2'  # automatic choose CUDA or CPU
                          ).load_trained_model()
