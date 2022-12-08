# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import AspectPolarityClassification as APC, ModelSaveOption, DeviceTypeOption
import warnings

warnings.filterwarnings('ignore')

for dataset in [
    APCDatasetList.Laptop14,
    APCDatasetList.Restaurant14,
    APCDatasetList.Restaurant15,
    APCDatasetList.Restaurant16,
    APCDatasetList.MAMS
]:
    for model in [
        APC.APCModelList.FAST_LSA_T_V2,
        APC.APCModelList.FAST_LSA_S_V2,
        # APC.APCModelList.BERT_SPC_V2
    ]:
        config = APC.APCConfigManager.get_apc_config_english()
        config.model = model
        config.pretrained_bert = 'microsoft/deberta-v3-base'
        # config.pretrained_bert = 'bert-base-uncased'
        config.evaluate_begin = 5
        config.max_seq_len = 80
        config.num_epoch = 30
        config.log_step = -1
        config.dropout = 0
        config.cache_dataset = False
        config.l2reg = 1e-8
        config.lsa = True
        # config.use_amp = True
        config.seed = [random.randint(0, 10000) for _ in range(3)]

        APC.APCTrainer(config=config,
                       dataset=dataset,
                       # from_checkpoint='english',
                       checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                       auto_device=DeviceTypeOption.ALL_CUDA,
                       ).destroy()
