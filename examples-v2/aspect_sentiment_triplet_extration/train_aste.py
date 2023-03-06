# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import (
    ModelSaveOption,
    DeviceTypeOption,
    DatasetItem,
)

import pyabsa.tasks.AspectSentimentTripletExtraction as ASTE

config = ASTE.ASTEConfigManager.get_aste_config_english()
config.max_seq_len = 80
config.log_step = -1
config.num_epoch = 1

dataset = "Laptop14"
trainer = ASTE.ASTETrainer(
    config=config,
    dataset=dataset,
    # from_checkpoint='english',
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    # checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
    auto_device=DeviceTypeOption.AUTO,
)
triplet_extractor = trainer.load_trained_model()

examples = [
    "I love this laptop, it is very good.",
    "I hate this laptop, it is very bad.",
    "I like this laptop, it is very good.",
    "I dislike this laptop, it is very bad.",
]
for example in examples:
    triplet_extractor.predict("I love this laptop, it is very good.")
