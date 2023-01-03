# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

import autocuda

from pyabsa.tasks.AspectPolarityClassification import APCDatasetList

from pyabsa import AspectPolarityClassification as APC, DeviceTypeOption
from pyabsa.augmentation import auto_aspect_sentiment_classification_augmentation
import warnings

from pyabsa.augmentation import auto_classification_augmentation

from pyabsa import TextClassification as TC

warnings.filterwarnings("ignore")


def test_classification_augmentation():
    config = TC.TCConfigManager.get_tc_config_english()
    config.model = TC.BERTTCModelList.BERT_MLP
    config.num_epoch = 1
    config.evaluate_begin = 0
    config.max_seq_len = 10
    config.dropout = 0.5
    config.seed = {42}
    config.log_step = -1
    config.l2reg = 0.00001
    config.data_num = 6

    SST2 = TC.TCDatasetList.SST2

    auto_classification_augmentation(
        config=config, dataset=SST2, device=DeviceTypeOption.AUTO
    )


def test_aspect_sentiment_classification_augmentation():
    for dataset in [
        APCDatasetList.Laptop14,
        # APCDatasetList.Restaurant14,
        # APCDatasetList.Restaurant15,
        # APCDatasetList.Restaurant16,
        # APCDatasetList.MAMS
    ]:
        for model in [
            APC.APCModelList.FAST_LSA_T_V2,
            # APC.APCModelList.FAST_LSA_S_V2,
            # APC.APCModelList.BERT_SPC_V2
        ]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.model = model
            config.pretrained_bert = "microsoft/deberta-v3-base"
            config.evaluate_begin = 5
            config.max_seq_len = 10
            config.num_epoch = 30
            config.log_step = 10
            config.dropout = 0
            config.cache_dataset = False
            config.l2reg = 1e-8
            config.lsa = True
            config.data_num = 6

            config.seed = [random.randint(0, 10000) for _ in range(1)]

            auto_aspect_sentiment_classification_augmentation(
                config=config, dataset=dataset, device=autocuda.auto_cuda()
            )


if __name__ == "__main__":
    test_classification_augmentation()
    test_aspect_sentiment_classification_augmentation()
