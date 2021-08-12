# -*- coding: utf-8 -*-
# file: train_apc_without_test_set.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import APCTrainer, ABSADatasetList, APCConfigManager

config = APCConfigManager.get_apc_config_base()

multilingual = ABSADatasetList.Multilingual

sent_classifier = APCTrainer(config=config,  # set config=None to use default model
                             dataset=multilingual,  # file or dir, dataset_utils(s) will be automatically detected
                             save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                             auto_device=True  # Auto choose CUDA or CPU
                             )
