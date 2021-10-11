# -*- coding: utf-8 -*-
# file: train_atepc_based_on_checkpoint.py
# time: 2021/7/27
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import ATEPCCheckpointManager
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

config = ATEPCConfigManager.get_atepc_config_english()

checkpoint_path = ATEPCCheckpointManager.get_checkpoint(checkpoint='checkpoint')  # or

config.model = ATEPCModelList.LCFS_ATEPC
config.evaluate_begin = 0
config.num_epoch = 1

TShirt = ABSADatasetList.TShirt
aspect_extractor = Trainer(config=config,
                           dataset=TShirt,
                           from_checkpoint=checkpoint_path,
                           checkpoint_save_mode=1,
                           auto_device=True
                           )
