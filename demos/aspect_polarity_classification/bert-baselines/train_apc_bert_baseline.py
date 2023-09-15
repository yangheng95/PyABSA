# -*- coding: utf-8 -*-
# file: train_apc_bert_baseline.py
# time: 2021/7/27
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import (
    APCTrainer,
    APCConfigManager,
    BERTBaselineAPCModelList,
    ABSADatasetList,
)

apc_config_english = APCConfigManager.get_apc_config_bert_baseline()
apc_config_english.model = BERTBaselineAPCModelList.ASGCN_BERT
apc_config_english.num_epoch = 10
apc_config_english.evaluate_begin = 2
apc_config_english.max_seq_len = 100
apc_config_english.dropout = 0.5
apc_config_english.log_step = 5
apc_config_english.l2reg = 0.0005
apc_config_english.seed = 1
apc_config_english.use_syntax_based_SRD = True
apc_config_english.similarity_threshold = 1
apc_config_english.cross_validate_fold = -1  # disable cross_validate

laptop14 = ABSADatasetList.Laptop14
sent_classifier = APCTrainer(
    config=apc_config_english,
    dataset=laptop14,  # train set and test set will be automatically detected
    checkpoint_save_mode=1,  # None to avoid save model
    auto_device=True,  # automatic choose CUDA or CPU
).train()

Restaurant14 = ABSADatasetList.Restaurant14
sent_classifier = APCTrainer(
    config=apc_config_english,
    dataset=Restaurant14,  # train set and test set will be automatically detected
    checkpoint_save_mode=1,  # None to avoid save model
    auto_device=True,  # automatic choose CUDA or CPU
).train()

Restaurant15 = ABSADatasetList.Restaurant15
sent_classifier = APCTrainer(
    config=apc_config_english,
    dataset=Restaurant15,  # train set and test set will be automatically detected
    checkpoint_save_mode=1,  # None to avoid save model
    auto_device=True,  # automatic choose CUDA or CPU
).train()

Restaurant16 = ABSADatasetList.Restaurant16
sent_classifier = APCTrainer(
    config=apc_config_english,
    dataset=Restaurant16,  # train set and test set will be automatically detected
    checkpoint_save_mode=1,  # None to avoid save model
    auto_device=True,  # automatic choose CUDA or CPU
).train()
