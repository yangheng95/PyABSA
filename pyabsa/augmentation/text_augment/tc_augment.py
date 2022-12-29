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

from termcolor import colored

from pyabsa.utils.pyabsa_utils import fprint


def auto_classification_augmentation(config, dataset, device, **kwargs):
    fprint(
        colored(
            "Performing augmentation for text classification. This may take a long time",
            "yellow",
        )
    )

    from pyabsa.tasks.TextClassification import BERTTCModelList
    from boost_aug import TCBoostAug

    config.model = BERTTCModelList.BERT_MLP

    augmentor = TCBoostAug(
        ROOT=os.getcwd(),
        BOOSTING_FOLD=kwargs.get("boosting_fold", 4),
        CLASSIFIER_TRAINING_NUM=kwargs.get("classifier_training_num", 1),
        WINNER_NUM_PER_CASE=kwargs.get("winner_num_per_case", 5),
        AUGMENT_NUM_PER_CASE=kwargs.get("augment_num_per_case", 10),
        device=device,
    )

    augmentor.tc_boost_augment(
        config,
        dataset,
        train_after_aug=kwargs.get("train_after_aug", True),
        rewrite_cache=kwargs.get("rewrite_cache", True),
    )
    sys.exit(0)
