# -*- coding: utf-8 -*-
# file: metric_visualization.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                          This code is for paper:                                                     #
#      "Back to Reality: Leveraging Pattern-driven Modeling to Enable Affordable Sentiment Dependency Learning"        #
#                      but there are some changes in this paper, and it is under submission                            #
########################################################################################################################
import warnings

import random

from metric_visualizer import MetricVisualizer

from pyabsa import TextClassificationTrainer, ClassificationConfigManager, ClassificationDatasetList
from pyabsa.functional import BERTClassificationModelList, Trainer

warnings.filterwarnings('ignore')
seeds = [random.randint(0, 10000) for _ in range(3)]


classification_config_english = ClassificationConfigManager.get_classification_config_english()
classification_config_english.model = BERTClassificationModelList.BERT
classification_config_english.num_epoch = 10
classification_config_english.evaluate_begin = 0
classification_config_english.max_seq_len = 80
classification_config_english.log_step = 200
classification_config_english.dropout = 0.5
classification_config_english.cache_dataset = True
classification_config_english.seed = {42, 56, 1}
classification_config_english.l2reg = 1e-8

max_seq_lens = [60, 70, 80, 90, 100]

MV = MetricVisualizer()
classification_config_english.MV = MV
for msl in max_seq_lens:
    classification_config_english.max_seq_len = msl
    Laptop14 = ClassificationDatasetList.SST2
    Trainer(config=classification_config_english,
            dataset=Laptop14,  # train set and test set will be automatically detected
            checkpoint_save_mode=0,  # =None to avoid save model
            auto_device=True  # automatic choose CUDA or CPU
            )
    classification_config_english.MV.next_trail()

save_path = '{}_{}'.format(classification_config_english.model_name, classification_config_english.dataset_name)
MV.summary(save_path=None)
MV.traj_plot(save_path=None)
MV.violin_plot(save_path=None)
MV.box_plot(save_path=None)

try:
    MV.summary(save_path=save_path)
    MV.traj_plot(save_path=save_path)
    MV.violin_plot(save_path=save_path)
    MV.box_plot(save_path=save_path)
except Exception as e:
    pass
