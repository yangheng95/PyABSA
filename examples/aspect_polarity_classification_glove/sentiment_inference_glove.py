# -*- coding: utf-8 -*-
# project: PyABSA
# file: sentiment_inference_glove.py
# time: 2021/7/23
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc, apc_config_handler

from pyabsa.model_utils import APCModelList
from pyabsa import ABSADatasets

########################################################################################################################
#                To use GloVe-based models, you should put the GloVe embedding into the dataset path                   #
#              or if you can access to Google, it will automatic download GloVe embedding if necessary                 #
########################################################################################################################

from pyabsa import load_sentiment_classifier
from pyabsa import ABSADatasets

# Assume the sent_classifier is loaded or obtained using train function

sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

# model_path = APCTrainedModelManager.get_checkpoint(checkpoint_name='English')
model_path = 'state_dict/tnet_lf_cdw_acc_81.07_f1_71.95'
sent_classifier = load_sentiment_classifier(trained_model_path=model_path,
                                            auto_device=True,  # Use CUDA if available
                                            sentiment_map=sentiment_map
                                            )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasets.SemEval
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
