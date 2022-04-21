# -*- coding: utf-8 -*-
# project: PyABSA
# file: train_apc_glove.py
# time: 2021/7/18
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import APCTrainer, APCConfigManager, GloVeAPCModelList, ABSADatasetList

# Put glove embedding under current path first if you dont want to download GloVe embedding
apc_config_english = APCConfigManager.get_apc_config_glove()
apc_config_english.model = GloVeAPCModelList.TNet_LF
apc_config_english.num_epoch = 20
apc_config_english.cross_validate_fold = -1  # disable cross_validate, enable in {5, 10}

Dataset = ABSADatasetList.SemEval
sent_classifier = APCTrainer(config=apc_config_english,  # set config=None will use the apc_config as well
                             dataset=Dataset,  # train set and test set will be automatically detected
                             checkpoint_save_mode=1,  # set model_path_to_save=None to avoid save model
                             auto_device=True  # automatic choose CUDA or CPU
                             ).train()
