# -*- coding: utf-8 -*-
# file: sentiment_inference.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
# Usage: Evaluate on given text or inference dataset_utils
import os

from pyabsa import APCCheckpointManager, ABSADatasetList

os.environ['PYTHONIOENCODING'] = 'UTF8'

# Assume the sent_classifier is loaded or obtained using train function

# sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='slide_lcfs_bert_acc_89.76_f1_69.44',
#                                                                 auto_device=True,  # Use CUDA if available
#                                                                 sentiment_map=sentiment_map
#                                                                 )

sent_classifier = APCCheckpointManager.get_sentiment_classifier(checkpoint='fast_lcf_bert_acc_77.9_f1_74.34',
                                                                auto_device=True,  # Use CUDA if available
                                                                )

# text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent , the [ASP]decor[ASP] cool and understated . !sent! 1 1'
# sent_classifier.infer(text, print_result=True)

# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ABSADatasetList.Laptop14
# inference_sets = r'E:\PyABSA-Workspace\latest\PyABSA\examples\aspect_polarity_classification\datasets\apc_datasets\mams'
results = sent_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=False,
                                      )
