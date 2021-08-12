# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import TextClassificationTrainer, ClassificationConfigManager, ClassificationDatasetList
from pyabsa.functional import BERTClassificationModelList

classification_config_english = ClassificationConfigManager.get_classification_config_english()
classification_config_english.model = BERTClassificationModelList.BERT
classification_config_english.num_epoch = 10
classification_config_english.evaluate_begin = 2
classification_config_english.max_seq_len = 80
classification_config_english.dropout = 0.5
classification_config_english.seed = {42, 56, 1}
classification_config_english.log_step = 5
classification_config_english.l2reg = 0.0001

SST2 = ClassificationDatasetList.SST2
text_classifier = TextClassificationTrainer(config=classification_config_english,  # set config=None to use default model
                                            dataset=SST2,  # train set and test set will be automatically detected
                                            save_checkpoint=True,  # set model_path_to_save=None to avoid save model
                                            auto_device=True  # automatic choose CUDA or CPU
                                            )
