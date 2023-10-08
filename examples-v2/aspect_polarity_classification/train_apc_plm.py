# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa import AspectPolarityClassification, ModelSaveOption

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

config = AspectPolarityClassification.APCConfigManager.get_apc_config_english()
config.evaluate_begin = 0
config.num_epoch = 1
config.max_seq_len = 160
config.log_step = -1
config.dropout = 0
config.l2reg = 1e-5
config.cache_dataset = False
config.seed = random.randint(0, 10000)
config.model = AspectPolarityClassification.BERTBaselineAPCModelList.ASGCN_BERT
# configuration_class.spacy_model = 'zh_core_web_sm'
# chinese_sets = ABSADatasetList.Chinese
chinese_sets = AspectPolarityClassification.APCDatasetList.ARTS_Laptop14
# chinese_sets = AspectPolarityClassification.APCDatasetList.ARTS_Restaurant14
# chinese_sets = ABSADatasetList.MOOC
sent_classifier = AspectPolarityClassification.APCTrainer(
    config=config,
    # set configuration_class=None to use default model
    dataset=chinese_sets,
    # train set and test set will be automatically detected
    checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
    auto_device=True,  # automatic choose CUDA or CPU
).load_trained_model()

from pyabsa import AspectPolarityClassification as APC

inference_sets = APC.APCDatasetList.Laptop14
results = sent_classifier.batch_predict(
    target_file=inference_sets,
    print_result=True,
    save_result=True,
    ignore_error=False,
)
