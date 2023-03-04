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


class AugmentBackend:
    EDA = "EDA"
    ContextualWordEmbsAug = "ContextualWordEmbsAug"
    RandomWordAug = "RandomWordAug"
    AntonymAug = "AntonymAug"
    SynonymAug = "SynonymAug"
    SplitAug = "SplitAug"
    BackTranslationAug = "BackTranslationAug"
    SpellingAug = "SpellingAug"


def auto_classification_augmentation(
    config,
    dataset,
    device: str,
    boosting_fold: int = 4,
    classifier_training_num: int = 1,
    augment_num_per_case: int = 10,
    winner_num_per_case: int = 5,
    augment_backend: str = "eda",
    train_after_aug: bool = True,
    rewrite_cache: bool = True,
) -> None:
    """
    Augment the dataset using BoostTextAugmentation tool (https://github.com/yangheng95/BoostTextAugmentation) for text
    classification.

    Args:
        config (ABSAConfig): The configuration object for ABSA.
        dataset (ABSADataset): The dataset to be augmented.
        device (str): The device to run the augment on.
        boosting_fold (int, optional): The number of boosting fold. Defaults to 4.
        classifier_training_num (int, optional): The number of classifier training. Defaults to 1.
        augment_num_per_case (int, optional): The number of augmented samples to generate per case. Defaults to 10.
        winner_num_per_case (int, optional): The number of winners per case. Defaults to 5.
        augment_backend (str, optional): The data augment backend to use. Defaults to "eda".
        train_after_aug (bool, optional): Whether to train the model after the data augmentation. Defaults to True.
        rewrite_cache (bool, optional): Whether to rewrite the cache files. Defaults to True.

    Returns:
        None
    """
    fprint(
        colored(
            "Performing augmentation for text classification. This may take a long time",
            "yellow",
        )
    )

    from pyabsa.tasks.TextClassification import BERTTCModelList
    from boost_aug import TCBoostAug, AugmentBackend

    config.model = BERTTCModelList.BERT_MLP

    augmentor = TCBoostAug(
        ROOT=os.getcwd(),
        BOOSTING_FOLD=boosting_fold,
        CLASSIFIER_TRAINING_NUM=classifier_training_num,
        AUGMENT_NUM_PER_CASE=augment_num_per_case,
        WINNER_NUM_PER_CASE=winner_num_per_case,
        AUGMENT_BACKEND=augment_backend,
        device=device,
    )

    augmentor.tc_boost_augment(
        config=config,
        dataset=dataset,
        train_after_aug=train_after_aug,
        rewrite_cache=rewrite_cache,
    )
