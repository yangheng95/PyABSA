# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

SemEval = ABSADatasetList.TShirt
atepc_config_english = ATEPCConfigManager.get_atepc_config_english()
atepc_config_english.num_epoch = 10
atepc_config_english.evaluate_begin = 4
atepc_config_english.lot_step = 100
atepc_config_english.model = ATEPCModelList.LCF_ATEPC
aspect_extractor = ATEPCTrainer(config=atepc_config_english,  # set config=None to use default model
                                dataset=SemEval,  # file or dir, dataset_utils(s) will be automatically detected
                                save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                                auto_device=True  # Auto choose CUDA or CPU
                                )
