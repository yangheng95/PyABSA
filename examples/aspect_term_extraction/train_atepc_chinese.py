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

chinese_sets = ABSADatasetList.Chinese
atepc_config_chinese = ATEPCConfigManager.get_atepc_config_chinese()
atepc_config_chinese.log_step = 50
atepc_config_chinese.model = ATEPCModelList.LCF_ATEPC
atepc_config_chinese.evaluate_begin = 5

aspect_extractor = Trainer(config=atepc_config_chinese,  # set config=None to use default model
                           dataset=chinese_sets,  # file or dir, dataset_utils(s) will be automatically detected
                           save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                           auto_device=True  # Auto choose CUDA or CPU
                           )
