# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################
import random
from pyabsa import (
    AspectPolarityClassification as APC,
    ModelSaveOption,
    DeviceTypeOption,
    DatasetItem,
)

models = [
    APC.APCModelList.FAST_LSA_T_V2,
    APC.APCModelList.FAST_LSA_S_V2,
    APC.APCModelList.BERT_SPC_V2,
]

datasets = DatasetItem(
    [
        APC.APCDatasetList.Laptop14,
        APC.APCDatasetList.Restaurant14,
        APC.APCDatasetList.Restaurant15,
        APC.APCDatasetList.Restaurant16,
        APC.APCDatasetList.MAMS,
    ]
)

for dataset in [
    APC.APCDatasetList.Laptop14,
    APC.APCDatasetList.Restaurant14,
    APC.APCDatasetList.Restaurant15,
    APC.APCDatasetList.Restaurant16,
    # APCDatasetList.MAMS
]:
    for model in [
        APC.APCModelList.FAST_LSA_T_V2,
        APC.APCModelList.FAST_LSA_S_V2,
        APC.APCModelList.BERT_SPC_V2,
        # APC.APCModelList.BERT_SPC
    ]:
        for pretrained_bert in [
            "microsoft/deberta-v3-base",
            # 'roberta-base',
            # 'microsoft/deberta-v3-large',
        ]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.model = model
            config.pretrained_bert = pretrained_bert
            # config.pretrained_bert = 'roberta-base'
            config.evaluate_begin = 0
            config.max_seq_len = 80
            config.num_epoch = 30
            config.log_step = 5
            # config.log_step = -1
            config.dropout = 0.5
            config.eta = -1
            config.eta_lr = 0.001
            # config.lcf = 'fusion'
            config.cache_dataset = False
            config.l2reg = 1e-8
            config.learning_rate = 1e-5
            config.use_amp = True
            config.use_bert_spc = True
            config.lsa = True
            config.use_torch_compile = False
            config.seed = [random.randint(0, 10000) for _ in range(3)]

            trainer = APC.APCTrainer(
                config=config,
                dataset=dataset,
                # from_checkpoint='english',
                checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                # checkpoint_save_mode=ModelSaveOption.DO_NOT_SAVE_MODEL,
                auto_device=DeviceTypeOption.AUTO,
            )
            trainer.load_trained_model()
