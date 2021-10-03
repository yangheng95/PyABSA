# -*- coding: utf-8 -*-
# file: 2_inference.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset_utils

from pyabsa import APCCheckpointManager, ABSADatasetList

# Assume the sent_classifier is loaded or obtained using train function

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='ASGCN_BERT',
                                                                auto_device=True,  # Use CUDA if available
                                                                sentiment_map=None
                                                                )

text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasetList.Laptop14

results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
