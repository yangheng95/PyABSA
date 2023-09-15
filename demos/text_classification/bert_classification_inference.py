# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

from pyabsa import TextClassifierCheckpointManager, ClassificationDatasetList

os.environ["PYTHONIOENCODING"] = "UTF8"

# Assume the text_classifier is loaded or obtained using train function

text_classifier = TextClassifierCheckpointManager.get_text_classifier(
    checkpoint="bert",
    auto_device=True,  # Use CUDA if available
    # cal_perplexity=True,
)

# batch inference returns the results, save the result if necessary using save_result=True
inference_sets = ClassificationDatasetList.SST2
results = text_classifier.batch_infer(
    target_file=inference_sets,
    print_result=True,
    save_result=False,
    ignore_error=False,
)
print(results)
