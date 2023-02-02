# -*- coding: utf-8 -*-
# file: test_10_apc_inference.py
# time: 12:07 2023/2/2
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import download_all_available_datasets, AspectPolarityClassification as APC, TaskCodeOption

import os
import shutil


def test_multilingual_inference():
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')

    download_all_available_datasets()
    classifier = APC.SentimentClassifier('multilingual')
    assert classifier.predict('The [B-ASP]food[E-ASP] is good', pred_sentiment=True)['sentiment'] == ['Positive']
    assert classifier.predict('The [B-ASP]food[E-ASP] is bad', pred_sentiment=True)['sentiment'] == ['Negative']
    assert classifier.predict('The [B-ASP]food[E-ASP] is not good', pred_sentiment=True)['sentiment'] == ['Negative']
    assert classifier.predict('The [B-ASP]food[E-ASP] is not bad', pred_sentiment=True)['sentiment'] == ['Neutral']
    # assert classifier.predict('The food is good', pred_sentiment=True)['sentiment'] == ['Positive']

    assert classifier.predict('La [B-ASP]comida[E-ASP] es buena', pred_sentiment=True)['sentiment'] == ['Positive']
    assert classifier.predict('La [B-ASP]comida[E-ASP] es mala', pred_sentiment=True)['sentiment'] == ['Negative']


def test_en_inference():
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')

    download_all_available_datasets()
    classifier = APC.SentimentClassifier('english')
    assert classifier.predict('The [B-ASP]food[E-ASP] is good', pred_sentiment=True)['sentiment'] == ['Positive']
    assert classifier.predict('The [B-ASP]food[E-ASP] is bad', pred_sentiment=True)['sentiment'] == ['Negative']
    assert classifier.predict('The [B-ASP]food[E-ASP] is not good', pred_sentiment=True)['sentiment'] == ['Negative']


def test_chinese_inference():
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')

    download_all_available_datasets()
    classifier = APC.SentimentClassifier('chinese')
    assert classifier.predict('[B-ASP]食物[E-ASP]很好', pred_sentiment=True)['sentiment'] == ['Positive']
    assert classifier.predict('[B-ASP]食物[E-ASP]很差', pred_sentiment=True)['sentiment'] == ['Negative']
    assert classifier.predict('[B-ASP]食物[E-ASP]不好', pred_sentiment=True)['sentiment'] == ['Negative']
    assert classifier.predict('[B-ASP]食物[E-ASP]不差', pred_sentiment=True)['sentiment'] == ['Positive']


if __name__ == '__main__':
    test_multilingual_inference()
    test_en_inference()
    test_chinese_inference()
