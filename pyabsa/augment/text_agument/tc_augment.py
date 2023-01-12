# -*- coding: utf-8 -*-
# file: tc_augment.py
# time: 02/11/2022 19:51
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import sys

import autocuda
from termcolor import colored


def auto_classification_augmentation(dataset, config, **kwargs):
    print(
        colored(
            "No augment datasets found, performing TC augment. this may take a long time...",
            "yellow",
        )
    )

    from boost_aug import TCBoostAug
    from pyabsa.tasks.TextClassification.configuration.tc_configuration import (
        TCConfigManager,
    )

    tc_config = TCConfigManager.get_classification_config_english()
    tc_config.log_step = -1

    BoostingAugmenter = TCBoostAug(
        ROOT=os.getcwd(),
        CLASSIFIER_TRAINING_NUM=kwargs.get("classifier_training_num", 1),
        WINNER_NUM_PER_CASE=kwargs.get("winner_num_per_case", 5),
        AUGMENT_NUM_PER_CASE=kwargs.get("augment_num_per_case", 10),
        device=config.device,
    )

    # auto-trainer after augment
    BoostingAugmenter.tc_boost_augment(
        tc_config,
        dataset,
        train_after_aug=kwargs.get("train_after_aug", True),
        rewrite_cache=kwargs.get("rewrite_cache", True),
    )
    sys.exit(0)
