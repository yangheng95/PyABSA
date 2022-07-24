# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import TCTrainer, TCConfigManager, TCDatasetList
config = TCConfigManager.get_tc_config_english()
config.cross_validate_fold = 5

dataset = TCDatasetList.SST2
text_classifier = TCTrainer(config=config,
                            dataset=dataset,
                            checkpoint_save_mode=1,
                            auto_device=True
                            ).load_trained_model()
