# -*- coding: utf-8 -*-
# file: text_classification_glove.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

from pyabsa import load_text_classifier, ClassificationDatasetList

os.environ['PYTHONIOENCODING'] = 'UTF8'

# Assume the text_classifier is loaded or obtained using train function


model_path = 'state_dict/lstm_acc_85.78_f1_85.77'
text_classifier = load_text_classifier(trained_model_path=model_path,
                                       auto_device=True,  # Use CUDA if available
                                       )


# batch inferring_tutorials returns the results, save the result if necessary using save_result=True
inference_sets = ClassificationDatasetList.SST2
results = text_classifier.batch_infer(target_file=inference_sets,
                                      print_result=True,
                                      save_result=True,
                                      ignore_error=True,
                                      )
