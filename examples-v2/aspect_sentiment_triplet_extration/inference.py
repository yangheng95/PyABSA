# -*- coding: utf-8 -*-
# file: inference.py
# time: 17:08 2023/3/6
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import AspectSentimentTripletExtraction as ASTE

# Load the model
# triplet_extractor = ASTE.AspectSentimentTripletExtractor("english")
triplet_extractor = ASTE.AspectSentimentTripletExtractor("multilingual")

# # Predict
examples = [
    "I would like to have volume buttons rather than the adjustment that is on the front .####[([5, 6], [2], 'NEG')]",
    "It runs perfectly .####[([1], [2], 'POS')]",
    "Sometimes the screen even goes black on this computer .####[([2], [5], 'NEG')]",
    "Its fast and another thing I like is that it has three USB ports .####[([12, 13], [6], 'POS')]",
]
for example in examples:
    triplet_extractor.predict(example)

# Batch predict
target_file = "ASTE"

triplet_extractor.batch_predict(
    target_file=target_file,
    batch_size=32,
    ignore_error=True,
    save_result=True,
    auto_device=True,
)
