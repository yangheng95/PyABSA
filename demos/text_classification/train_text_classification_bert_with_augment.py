# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

import autocuda
from boost_aug import BoostingAug, AugmentBackend

from pyabsa import TextClassificationTrainer, ClassificationConfigManager, ClassificationDatasetList
from pyabsa.functional import BERTClassificationModelList

device = autocuda.auto_cuda()
aug_backend = AugmentBackend.EDA
# AugmentBackend.SplitAug
# AugmentBackend.SpellingAug
# AugmentBackend.ContextualWordEmbsAug
# AugmentBackend.BackTranslationAug
classification_config_english = ClassificationConfigManager.get_classification_config_english()
classification_config_english.model = BERTClassificationModelList.BERT
classification_config_english.num_epoch = 10
classification_config_english.evaluate_begin = 0
classification_config_english.max_seq_len = 512
classification_config_english.log_step = 200
classification_config_english.dropout = 0.5
classification_config_english.cache_dataset = False
classification_config_english.seed = {42, 56, 1}
classification_config_english.l2reg = 1e-5
classification_config_english.learning_rate = 1e-5
classification_config_english.cross_validate_fold = 5

dataset = ClassificationDatasetList.SST2

# Our data augmentation tool © can automatically improve your dataset's performance 1-2% with additional computation budget
# Note to use our augmentation method, please put your dataset in integrated_datasets folder, and we encourage you to
# share your dataset at https://github.com/yangheng95/ABSADatasets, all the copyrights belong to the owner according to the licence

BoostingAugmenter = BoostingAug(ROOT=os.getcwd(),
                                AUGMENT_BACKEND=aug_backend,
                                WINNER_NUM_PER_CASE=8,
                                AUGMENT_NUM_PER_CASE=15,
                                device=device)
sent_classifier = BoostingAugmenter.apc_cross_boost_training(classification_config_english,
                                                             dataset,
                                                             rewrite_cache=True,
                                                             train_after_aug=False
                                                             )
