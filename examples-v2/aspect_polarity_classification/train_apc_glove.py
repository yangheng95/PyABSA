# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import ModelSaveOption, DeviceTypeOption
from pyabsa import AspectPolarityClassification as APC

for dataset in [APC.APCDatasetList.Laptop14]:
    for model in APC.GloVeAPCModelList():
        config = APC.APCConfigManager.get_apc_config_glove()
        config.lcf = 'cdm'
        config.model = APC.GloVeAPCModelList.TNet_LF
        config.cache_dataset = True
        config.overwrite_cache = True
        config.num_epoch = 10
        config.max_seq_len = 512
        config.evaluate_begin = 0
        config.log_step = -1
        config.cross_validate_fold = -1
        sent_classifier = APC.APCTrainer(config=config,
                                         dataset=dataset,
                                         checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                         auto_device=DeviceTypeOption.ALL_CUDA
                                         ).load_trained_model()


        inference_sets = APC.APCDatasetList.Laptop14
        results = sent_classifier.batch_predict(target_file=inference_sets,
                                                print_result=True,
                                                save_result=True,
                                                ignore_error=False,
                                                )

        sent_classifier.destroy()