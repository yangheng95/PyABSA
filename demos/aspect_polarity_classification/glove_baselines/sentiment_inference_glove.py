# -*- coding: utf-8 -*-
# project: PyABSA
# file: sentiment_inference_glove.py
# time: 2021/7/23
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os

from pyabsa import ABSADatasetList
from pyabsa import APCCheckpointManager

########################################################################################################################
#                To use GloVe-based model, you should put the GloVe embedding into the dataset path                   #
#              or if you can access to Google, it will automatic download GloVe embedding if necessary                 #
########################################################################################################################

os.environ['PYTHONIOENCODING'] = 'UTF8'

# Assume the sent_classifier is loaded or obtained using train function

# model_path = APCCheckpointManager.get_checkpoint(checkpoint_name='TNet_LF')
model_path = 'tnet_lf_acc_88.7_f1_64.02'  # test checkpoint_class
sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint=model_path,
                                                                auto_device=True,  # Use CUDA if available
                                                                sentiment_map=sentiment_map
                                                                )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.predict(text, print_result=True)

# batch inference returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasetList.SemEval
results = sent_classifier.batch_predict(target_file=inference_sets,
                                        print_result=True,
                                        save_result=True,
                                        ignore_error=True,
                                        )
