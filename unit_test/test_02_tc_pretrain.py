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

warnings.filterwarnings("ignore")

#######################################################################################################
#                                  This script is used for basic test                                 #
#                         The configuration_class test are ignored due to computation limitation                   #
#######################################################################################################

atepc_examples = [
    "But the staff was so nice to us .",
    "But the staff was so horrible to us .",
    r"Not only was the food outstanding , but the little ` perks \' were great .",
    "It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !",
    "It was pleasantly uncrowded , the service was delightful , the garden adorable , "
    "the food -LRB- from appetizers to entrees -RRB- was delectable .",
    "How pretentious and inappropriate for MJ Grill to claim that it provides power lunch and dinners !",
]

apc_examples = [
    "Strong build though which really adds to its [ASP]durability[ASP] .",  # $LABEL$ Positive
    "Strong [ASP]build[ASP] though which really adds to its durability . $LABEL$ Positive",
    "The [ASP]battery life[ASP] is excellent - 6-7 hours without charging . $LABEL$ Positive",
    "I have had my computer for 2 weeks already and it [ASP]works[ASP] perfectly . $LABEL$ Positive",
    "And I may be the only one but I am really liking [ASP]Windows 8[ASP] . $LABEL$ Positive",
]


def test_all_bert_models():
    from pyabsa import TextClassification as TC

    for model in TC.BERTTCModelList():
        cuda.empty_cache()
        config = TC.TCConfigManager.get_tc_config_english()
        config.model = model
        config.num_epoch = 1
        config.evaluate_begin = 0
        config.log_step = -1
        config.cache_dataset = True
        config.data_num = 6

        text_classifier = TC.TCTrainer(
            config=config,
            dataset=TC.TCDatasetList.SST2,
            checkpoint_save_mode=1,
            auto_device=DeviceTypeOption.AUTO,
        ).load_trained_model()
        text_classifier.predict("I love it very much!")


def test_all_glove_models():
    from pyabsa import TextClassification as TC

    for model in TC.GloVeTCModelList():
        config = TC.TCConfigManager.get_tc_config_glove()
        config.model = model
        config.num_epoch = 10
        config.num_epoch = 100
        config.evaluate_begin = 2
        config.max_seq_len = 10
        config.do_lower_case = True
        config.dropout = 0.5
        config.log_step = 5
        config.l2reg = 0.001
        config.data_num = 6

        text_classifier = TC.TCTrainer(
            config=config,
            dataset=TC.TCDatasetList.SST2,
            checkpoint_save_mode=1,
            auto_device=DeviceTypeOption.AUTO,
        ).load_trained_model()
        text_classifier.predict("I love it very much!")


if __name__ == "__main__":
    # test_all_bert_models()
    test_all_glove_models()
