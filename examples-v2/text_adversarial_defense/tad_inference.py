# -*- coding: utf-8 -*-
# file: bert_classification_inference.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os

import findfile
from pyabsa import TextAdversarialDefense as TAD, DatasetItem

os.environ["PYTHONIOENCODING"] = "UTF8"

# dataset = 'integrated_datasets/tc_datasets/201.SST2/stsa.binary.test.dat.adv'
# dataset = 'integrated_datasets/tc_datasets/204.AGNews10K/AGNews10K.test.dat.adv.0'
# dataset = 'integrated_datasets/tc_datasets/204.AGNews10K/AGNews10K.test.dat.adv'
dataset = (
    "integrated_datasets/tc_datasets/206.Amazon_Review_Polarity10K/amazon.test.dat.adv"
)
text_classifier = TAD.TADTextClassifier(
    # "TAD-SST2",
    # "TAD-AGNews10K",
    "TAD-Amazon",
    auto_device=True,  # Use CUDA if available
    # defense=None,
)
num_acc, num_all = 0, 0
lines = open(dataset, "r", encoding="utf-8").readlines()
for line in lines:
    # inference_sets = DatasetItem(dataset)
    line = json.loads(line)
    text, label = line["text"], line["label"]
    result = text_classifier.predict(
        text=text,
        print_result=False,
        save_result=False,
        ignore_error=False,
        defense="pwws",
    )
    if result["label"] == label:
        num_acc += 1
    num_all += 1

    print(f"{num_acc}/{num_all}={num_acc/num_all}")
