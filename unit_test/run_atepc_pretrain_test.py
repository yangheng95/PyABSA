# -*- coding: utf-8 -*-
# file: run_test.py
# time: 2021/12/4
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import shutil

from torch import cuda

from findfile import find_cwd_dir

import warnings

from pyabsa import DeviceTypeOption

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


def test_chinese_atepc_models():
    from pyabsa import AspectTermExtraction as ATEPC
    # # for dataset in ABSADatasetList():
    for dataset in ATEPC.ATEPCDatasetList.Phone:
        for model in ATEPC.ATEPCModelList():
            config = ATEPC.ATEPCConfigManager.get_atepc_config_chinese()
            cuda.empty_cache()
            config.model = model
            config.cache_dataset = True
            config.num_epoch = 1
            config.evaluate_begin = 0
            config.max_seq_len = 10
            config.log_step = -1
            config.ate_loss_weight = 5
            config.show_metric = -1
            config.output_dim = 3
            config.num_labels = 6
            trainer = ATEPC.ATEPCTrainer(config=config,
                                         dataset=dataset,
                                         checkpoint_save_mode=1,
                                         auto_device=DeviceTypeOption.ALL_CUDA,
                                         )
            aspect_extractor = trainer.load_trained_model()
            trainer.destroy()

            aspect_extractor.batch_predict(inference_source=atepc_examples,  #
                                           save_result=True,
                                           print_result=True,  # print the result
                                           pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                           )
            aspect_extractor.destroy()


def test_all_ate_models():
    from pyabsa import AspectTermExtraction as ATEPC
    # # for dataset in ABSADatasetList():
    for dataset in ATEPC.ATEPCDatasetList()[:1]:
        for model in ATEPC.ATEPCModelList():
            config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
            cuda.empty_cache()
            config.model = model
            config.cache_dataset = True
            config.num_epoch = 1
            config.evaluate_begin = 0
            config.max_seq_len = 10
            config.log_step = -1
            config.ate_loss_weight = 5
            config.show_metric = -1
            config.output_dim = 3
            config.num_labels = 6
            trainer = ATEPC.ATEPCTrainer(config=config,
                                         dataset=dataset,
                                         checkpoint_save_mode=1,
                                         auto_device=DeviceTypeOption.ALL_CUDA,
                                         )
            aspect_extractor = trainer.load_trained_model()
            trainer.destroy()

            aspect_extractor.batch_predict(inference_source=atepc_examples,  #
                                           save_result=True,
                                           print_result=True,  # print the result
                                           pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                           )
            aspect_extractor.destroy()
