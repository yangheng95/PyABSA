# -*- coding: utf-8 -*-
# project: PyABSA
# file: sentiment_inference_glove.py
# time: 2021/7/23
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

from pyabsa import ABSADatasetList
from pyabsa import APCCheckpointManager

########################################################################################################################
#                To use GloVe-based model, you should put the GloVe embedding into the dataset_utils path                   #
#              or if you can access to Google, it will automatic download GloVe embedding if necessary                 #
########################################################################################################################

os.environ['PYTHONIOENCODING'] = 'UTF8'

# Assume the sent_classifier is loaded or obtained using train function

sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}

# model_path = APCCheckpointManager.get_checkpoint(checkpoint_name='TNet_LF')
model_path = 'asgcn_acc_81.34_f1_73.15'
sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint=model_path,
                                                                auto_device=True,  # Use CUDA if available
                                                                sentiment_map=sentiment_map
                                                                )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasetList.Restaurant15
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
