# -*- coding: utf-8 -*-
# file: train_atepc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
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
config.model = ATEPCModelList.LCF_ATEPC
config.evaluate_begin = 2
config.num_epoch = 20
config.log_step = 100
config.pretrained_bert = 'bert-base-uncased'
semeval = ABSADatasetList.Restaurant14
aspect_extractor = Trainer(config=config,
                           dataset=semeval,
                           checkpoint_save_mode=0,
                           auto_device=True
                           ).load_trained_model()

aspect_extractor.extract_aspect(
    ['the wine list is incredible and extensive and diverse , the food is all incredible and the staff was all very nice , ood at their jobs and cultured .',
     'One night I turned the freaking thing off after using it , the next day I turn it on , no GUI , screen all dark , power light steady , hard drive light steady and not flashing as it usually does .']
)
