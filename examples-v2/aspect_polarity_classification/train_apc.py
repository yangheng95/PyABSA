# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/5/26 0026
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import random

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################

from pyabsa import AspectPolarityClassification as APC, ModelSaveOption, DeviceTypeOption

config = APC.APCConfigManager.get_apc_config_english()
config.evaluate_begin = 0
config.num_epoch = 1
config.log_step = -1
config.dropout = 0
config.l2reg = 1e-5
config.seed = random.randint(0, 10000)
config.model = APC.APCModelList.FAST_LCF_BERT
# configuration_class.spacy_model = 'zh_core_web_sm'
# chinese_sets = ABSADatasetList.Chinese
dataset = APC.APCDatasetList.Laptop14
# chinese_sets = ABSADatasetList.MOOC
sent_classifier = APC.APCTrainer(config=config,  # set configuration_class=None to use default model
                                 dataset=dataset,  # train set and test set will be automatically detected
                                 checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                 auto_device=DeviceTypeOption.AUTO  # automatic choose CUDA or CPU
                                 ).load_trained_model()

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1, 1'
sent_classifier.predict(text, print_result=True)
sent_classifier.batch_predict(dataset, print_result=True)
