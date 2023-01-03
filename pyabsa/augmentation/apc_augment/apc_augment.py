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

from pyabsa.utils.pyabsa_utils import fprint


def auto_aspect_sentiment_classification_augmentation(
    config, dataset, device, **kwargs
):
    fprint(
        colored(
            "Performing augmentation for aspect sentiment classification. This may take a long time",
            "yellow",
        )
    )

    fprint(
        colored(
            "The augment tool is available at: {}".format(
                "https://github.com/yangheng95/BoostTextAugmentation"
            ),
            "yellow",
        )
    )
    from pyabsa.tasks.AspectPolarityClassification import APCModelList
    from boost_aug import ABSCBoostAug

    config.model = APCModelList.FAST_LCF_BERT

    augmentor = ABSCBoostAug(
        ROOT=os.getcwd(),
        BOOSTING_FOLD=kwargs.get("boosting_fold", 4),
        CLASSIFIER_TRAINING_NUM=kwargs.get("classifier_training_num", 1),
        AUGMENT_NUM_PER_CASE=kwargs.get("augment_num_per_case", 10),
        WINNER_NUM_PER_CASE=kwargs.get("winner_num_per_case", 5),
        device=device,
    )

    augmentor.apc_boost_augment(
        config,  # BOOSTAUG
        dataset,
        train_after_aug=kwargs.get("train_after_aug", True),
        rewrite_cache=kwargs.get("rewrite_cache", True),
    )
