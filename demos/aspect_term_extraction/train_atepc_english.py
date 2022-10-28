# -*- coding: utf-8 -*-
# file: train_atepc_english.py
# time: 2021/6/8 0008
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

config = ATEPCConfigManager.get_atepc_config_english()
config.model = ATEPCModelList.FAST_LCF_ATEPC
config.evaluate_begin = 0
config.log_step = -1
config.batch_size = 16
config.num_epoch = 30
config.max_seq_len = 128
config.cache_dataset = False
config.use_bert_spc = True
config.l2reg = 1e-5
config.learning_rate = 1e-5
multilingual = ABSADatasetList.English
config.pretrained_bert = 'yangheng/deberta-v3-base-absa-v1.1'
Dataset = ABSADatasetList.English

aspect_extractor = Trainer(config=config,
                           dataset=Dataset,
                           checkpoint_save_mode=1,
                           auto_device=True,
                           load_aug=True
                           ).load_trained_model()

aspect_extractor.extract_aspect(
    ['the wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice , ood at their jobs and cultured .',
     'One night I turned the freaking thing off after using it , the next day I turn it on , no GUI , screen all dark , power light steady , hard drive light steady and not flashing as it usually does .']
)
