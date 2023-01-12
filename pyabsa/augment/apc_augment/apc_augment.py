# -*- coding: utf-8 -*-
# file: apc_augment.py
# time: 02/11/2022 19:51
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import sys

from termcolor import colored

from pyabsa.tasks.AspectPolarityClassification import APCConfigManager, APCModelList


def auto_aspect_sentiment_classification_augmentation(dataset, config, **kwargs):
    print(
        colored(
            "No augment datasets found, performing APC augment. This may take a long time...",
            "yellow",
        )
    )
    print(
        colored(
            "The augment tool is available at: {}".format(
                "https://github.com/yangheng95/BoostTextAugmentation"
            ),
            "yellow",
        )
    )
    from boost_aug import ABSCBoostAug

    config = APCConfigManager.get_apc_config_english()
    config.model = APCModelList.FAST_LCF_BERT

    BoostingAugmenter = ABSCBoostAug(
        ROOT=os.getcwd(),
        CLASSIFIER_TRAINING_NUM=kwargs.get("classifier_training_num", 1),
        AUGMENT_NUM_PER_CASE=kwargs.get("augment_num_per_case", 10),
        WINNER_NUM_PER_CASE=kwargs.get("winner_num_per_case", 5),
        device=config.device,
    )

    # auto-trainer after augment
    BoostingAugmenter.apc_boost_augment(
        config,  # BOOSTAUG
        dataset,
        train_after_aug=kwargs.get("train_after_aug", True),
        rewrite_cache=kwargs.get("rewrite_cache", True),
    )
    sys.exit(0)
