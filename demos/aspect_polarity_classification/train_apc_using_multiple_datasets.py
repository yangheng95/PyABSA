# -*- coding: utf-8 -*-
# file: train_apc_using_multiple_datasets.py
# time: 2021/6/4 0004
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa.framework import Trainer
from pyabsa.framework import APCConfigManager
from pyabsa.framework import ABSADatasetList
from pyabsa.framework import APCModelList

# You can place multiple atepc_datasets file in one dir to easily train using some atepc_datasets

# for example, training_tutorials on the SemEval atepc_datasets, you can organize the dir as follow

# ATEPC同样支持多数据集集成训练，但请不要将极性标签（种类，长度）不同的数据集融合训练！
# --atepc_datasets
# ----laptop14
# ----restaurant14
# ----restaurant15
# ----restaurant16

# or
# --atepc_datasets
# ----SemEval2014
# ------laptop14
# ------restaurant14
# ----SemEval2015
# ------restaurant15
# ----SemEval2016
# ------restaurant16


semeval = ABSADatasetList.SemEval
sent_classifier = Trainer(config=APCConfigManager.get_apc_config_english(),
                          dataset=semeval,
                          checkpoint_save_mode=1,
                          auto_device=True
                          ).load_trained_model()
