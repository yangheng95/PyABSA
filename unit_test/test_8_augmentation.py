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

from pyabsa import TextClassification as TC, download_all_available_datasets
import os
import shutil

warnings.filterwarnings("ignore")

# #
# def test_classification_augmentation():
#
#     if os.path.exists("integrated_datasets"):
#         shutil.rmtree("integrated_datasets")
#     download_all_available_datasets()
#
#     config = TC.TCConfigManager.get_tc_config_english()
#     config.model = TC.BERTTCModelList.BERT_MLP
#     # config.pretrained_bert = "bert-base-uncased"
#     config.num_epoch = 5
#     config.evaluate_begin = 0
#     config.max_seq_len = 3
#     config.dropout = 0.5
#     config.seed = {42}
#     config.log_step = -1
#     config.l2reg = 0.00001
#
#     auto_classification_augmentation(
#         config=config,
#         dataset=TC.TCDatasetList.SST2,
#         device=autocuda.auto_cuda(),
#         # boosting_fold=2,
#         # classifier_training_num=1,
#         # augment_num_per_case=2,
#         # winner_num_per_case=1,
#         boosting_fold=4,
#         classifier_training_num=1,
#         augment_num_per_case=10,
#         winner_num_per_case=5,
#         augment_backend="ContextualWordEmbsAug",
#     )
#
#
# def test_aspect_sentiment_classification_augmentation():
#
#     if os.path.exists("integrated_datasets"):
#         shutil.rmtree("integrated_datasets")
#     download_all_available_datasets()
#
#     config = APC.APCConfigManager.get_apc_config_english()
#     # config.pretrained_bert = "bert-base-uncased"
#     config.evaluate_begin = 0
#     config.max_seq_len = 10
#     config.num_epoch = 10
#     config.log_step = 10
#     config.dropout = 0
#     config.cache_dataset = False
#     config.l2reg = 1e-8
#     config.lsa = True
#
#     config.seed = [random.randint(0, 10000) for _ in range(1)]
#
#     auto_aspect_sentiment_classification_augmentation(
#         config=config,
#         dataset=APC.APCDatasetList.Restaurant16,
#         device=autocuda.auto_cuda(),
#         # boosting_fold=2,
#         # classifier_training_num=1,
#         # augment_num_per_case=2,
#         # winner_num_per_case=1,
#         boosting_fold=4,
#         classifier_training_num=1,
#         augment_num_per_case=10,
#         winner_num_per_case=5,
#         augment_backend="ContextualWordEmbsAug",
#     )


def test_classification_augmentation():

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")
    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    download_all_available_datasets()

    config = TC.TCConfigManager.get_tc_config_english()
    config.model = TC.BERTTCModelList.BERT_MLP
    config.pretrained_bert = "bert-base-uncased"
    config.num_epoch = 1
    config.evaluate_begin = 0
    config.max_seq_len = 3
    config.dropout = 0.5
    config.seed = {42}
    config.log_step = -1
    config.l2reg = 0.00001
    config.data_num = 20

    auto_classification_augmentation(
        config=config,
        dataset=TC.TCDatasetList.SST2,
        device=autocuda.auto_cuda(),
        boosting_fold=2,
        classifier_training_num=1,
        augment_num_per_case=2,
        winner_num_per_case=1,
        # boosting_fold=4,
        # classifier_training_num=1,
        # augment_num_per_case=10,
        # winner_num_per_case=5,
        augment_backend="EDA",
    )


def test_aspect_sentiment_classification_augmentation():

    if os.path.exists("integrated_datasets"):
        shutil.rmtree("integrated_datasets")
    if os.path.exists("source_datasets.backup"):
        shutil.rmtree("source_datasets.backup")

    download_all_available_datasets()

    config = APC.APCConfigManager.get_apc_config_english()
    config.pretrained_bert = "bert-base-uncased"
    config.evaluate_begin = 0
    config.max_seq_len = 10
    config.num_epoch = 1
    config.log_step = 10
    config.dropout = 0
    config.cache_dataset = False
    config.l2reg = 1e-8
    config.lsa = True
    config.data_num = 30

    config.seed = [random.randint(0, 10000) for _ in range(1)]

    auto_aspect_sentiment_classification_augmentation(
        config=config,
        dataset=APC.APCDatasetList.Restaurant14,
        device=autocuda.auto_cuda(),
        boosting_fold=2,
        classifier_training_num=1,
        augment_num_per_case=2,
        winner_num_per_case=1,
        # boosting_fold=4,
        # classifier_training_num=1,
        # augment_num_per_case=10,
        # winner_num_per_case=5,
        augment_backend="SynonymAug",
    )


if __name__ == "__main__":
    # test_classification_augmentation()
    test_aspect_sentiment_classification_augmentation()
