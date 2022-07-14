# -*- coding: utf-8 -*-
# file: train_atepc_multilingual.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training script                                                  #
########################################################################################################################

from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

config = ATEPCConfigManager.get_atepc_config_multilingual()
config.evaluate_begin = 0
config.log_step = -1
config.batch_size = 16
config.num_epoch = 30
config.max_seq_len = 256
config.cache_dataset = False
config.use_bert_spc = True
config.l2reg = 1e-5
# config.ate_loss_weight = 2  # (0, inf)
config.learning_rate = 1e-5
config.model = ATEPCModelList.FAST_LCF_ATEPC
multilingual = ABSADatasetList.Multilingual
config.pretrained_bert = 'microsoft/mdeberta-v3-base'

aspect_extractor = Trainer(config=config,
                           dataset=multilingual,
                           checkpoint_save_mode=1,
                           auto_device='allcuda',
                           load_aug=True
                           ).load_trained_model()
