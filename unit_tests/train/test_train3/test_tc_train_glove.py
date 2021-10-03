# -*- coding: utf-8 -*-
# file: 1_train_glove.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import ClassificationConfigManager, ClassificationDatasetList
from pyabsa.functional import GloVeClassificationModelList, Trainer

classification_config_english = ClassificationConfigManager.get_classification_config_glove()
classification_config_english.model = GloVeClassificationModelList.LSTM
classification_config_english.num_epoch = 10
classification_config_english.evaluate_begin = 2
classification_config_english.max_seq_len = 80
classification_config_english.dropout = 0.5
classification_config_english.seed = {42}
classification_config_english.log_step = 5
classification_config_english.l2reg = 0.0001

SST2 = ClassificationDatasetList.SST2
sent_classifier = Trainer(config=classification_config_english,
                          dataset=SST2,
                          checkpoint_save_mode=1,
                          auto_device=True
                          )
