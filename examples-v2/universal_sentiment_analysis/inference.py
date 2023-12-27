# -*- coding: utf-8 -*-
# file: inference.py
# time: 23:18 06/12/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.

from pyabsa import UniversalSentimentAnalysis as USA

usa_predictor = USA.USAPredictor(checkpoint="checkpoints")

examples = [
    '{"text": "keyboard key fragile .", "labels": [{"aspect": "keyboard", "opinion": "fragile", "polarity": "negative", "category": "KEYBOARD#QUALITY"}]}'
]

inference_results = usa_predictor.predict(examples)

print(inference_results)

while True:
    text = input("Please input your text: ")
    inference_results = usa_predictor.predict(text)
    print(inference_results)
