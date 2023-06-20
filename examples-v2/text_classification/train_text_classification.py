# -*- coding: utf-8 -*-
# file: train_text_classification_glove.py
# time: 2021/8/5
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


from pyabsa import TextClassification as TC

classification_config_english = TC.TCConfigManager.get_tc_config_english()
classification_config_english.model = TC.BERTTCModelList.BERT_MLP
classification_config_english.num_epoch = 20
classification_config_english.batch_size = 16
classification_config_english.evaluate_begin = 0
classification_config_english.max_seq_len = 100
classification_config_english.learning_rate = 5e-5
classification_config_english.dropout = 0
classification_config_english.seed = {42, 14, 5324}
classification_config_english.log_step = -1
classification_config_english.l2reg = 0.00001
# classification_config_english.use_amp = True
classification_config_english.cache_dataset = False

SST2 = "evoprompt"
sent_classifier = TC.TCTrainer(
    config=classification_config_english,
    dataset=SST2,
    checkpoint_save_mode=1,
    load_aug=False,
    auto_device=True,
).load_trained_model()
