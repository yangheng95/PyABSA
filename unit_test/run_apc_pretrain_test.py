# -*- coding: utf-8 -*-
# file: run_test.py
# time: 2021/12/4
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import shutil

from torch import cuda

from pyabsa import DeviceTypeOption, ModelSaveOption

import warnings

warnings.filterwarnings('ignore')

#######################################################################################################
#                                  This script is used for basic test                                 #
#                         The configuration_class test are ignored due to computation limitation                   #
#######################################################################################################

atepc_examples = ['But the staff was so nice to us .',
                  'But the staff was so horrible to us .',
                  r'Not only was the food outstanding , but the little ` perks \' were great .',
                  'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !',
                  'It was pleasantly uncrowded , the service was delightful , the garden adorable , '
                  'the food -LRB- from appetizers to entrees -RRB- was delectable .',
                  'How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !'
                  ]

apc_examples = [
    'Strong build though which really adds to its [ASP]durability[ASP] .',  # $LABEL$ Positive
    'Strong [ASP]build[ASP] though which really adds to its durability . $LABEL$ Positive',
    'The [ASP]battery life[ASP] is excellent - 6-7 hours without charging . $LABEL$ Positive',
    'I have had my computer for 2 weeks already and it [ASP]works[ASP] perfectly . $LABEL$ Positive',
    'And I may be the only one but I am really liking [ASP]Windows 8[ASP] . $LABEL$ Positive',
]


def test_cross_validate():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14, APC.APCDatasetList.Phone]:
        for model in APC.APCModelList()[:1]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.lcf = 'cdm'
            config.model = model
            config.cache_dataset = True
            config.num_epoch = 1
            config.max_seq_len = 10
            config.evaluate_begin = 0
            config.log_step = -1
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                             auto_device=DeviceTypeOption.ALL_CUDA
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()


def test_auto_device():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14, APC.APCDatasetList.Phone]:
        for model in APC.APCModelList()[:1]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.lcf = 'cdm'
            config.model = model
            config.cache_dataset = True
            config.cache_dataset = False
            config.num_epoch = 1
            config.max_seq_len = 10
            config.evaluate_begin = 0
            config.log_step = -1
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                             auto_device=DeviceTypeOption.AUTO
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()


def test_lcf_apc_models():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14]:
        for model in APC.APCModelList():
            config = APC.APCConfigManager.get_apc_config_english()
            config.lcf = 'cdm'
            config.model = model
            config.cache_dataset = True
            config.num_epoch = 1
            config.max_seq_len = 10
            config.evaluate_begin = 0
            config.log_step = -1
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                             auto_device=DeviceTypeOption.ALL_CUDA
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()


def test_save_models():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14, APC.APCDatasetList.Phone]:
        for model in APC.APCModelList()[:1]:
            config = APC.APCConfigManager.get_apc_config_english()
            config.lcf = 'cdm'
            config.model = model
            config.cache_dataset = True
            config.num_epoch = 1
            config.max_seq_len = 10
            config.evaluate_begin = 0
            config.log_step = -1
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=ModelSaveOption.SAVE_FULL_MODEL,
                                             auto_device=DeviceTypeOption.ALL_CUDA
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()


def test_bert_apc_models():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14, APC.APCDatasetList.Phone]:

        for model in APC.BERTBaselineAPCModelList():
            config = APC.APCConfigManager.get_apc_config_english()
            cuda.empty_cache()
            config.model = model
            config.cache_dataset = True
            config.max_seq_len = 128
            config.num_epoch = 1
            config.evaluate_begin = 0
            config.log_step = -1
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=2,
                                             auto_device=DeviceTypeOption.ALL_CUDA
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()


def test_glove_apc_models():
    from pyabsa import AspectPolarityClassification as APC

    for dataset in [APC.APCDatasetList.Laptop14]:
        for model in APC.GloVeAPCModelList():
            cuda.empty_cache()
            config = APC.APCConfigManager.get_apc_config_glove()
            config.model = model
            config.cache_dataset = True
            config.overwrite_cache = True
            config.num_epoch = 100
            config.patience = 20
            config.max_seq_len = 256
            config.evaluate_begin = 0
            config.log_step = 10
            sent_classifier = APC.APCTrainer(config=config,
                                             dataset=dataset,
                                             checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
                                             auto_device=DeviceTypeOption.ALL_CUDA
                                             ).load_trained_model()
            for ex in apc_examples:
                result = sent_classifier.predict(ex, print_result=True, ignore_error=False)

            sent_classifier.destroy()
