# -*- coding: utf-8 -*-
# file: inference.py
# time: 05/11/2022 19:48
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import random

import findfile
import tqdm
from sklearn.metrics import classification_report

from pyabsa import AspectPolarityClassification as APC


def ensemble_predict(apc_classifiers: dict, text, print_result=False):
    result = []
    for key, apc_classifier in apc_classifiers.items():
        result += apc_classifier.predict(text, print_result=print_result)['sentiment']
    return max(set(result), key=result.count)


def ensemble_performance(dataset, print_result=False):
    ckpts = findfile.find_cwd_dirs(dataset+'_acc')
    random.shuffle(ckpts)
    apc_classifiers = {}
    for ckpt in ckpts[:]:
        apc_classifiers[ckpt] = (APC.SentimentClassifier(ckpt))
    inference_file = {
        'laptop14': 'integrated_datasets/apc_datasets/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg.inference',
        'restaurant14': 'integrated_datasets/apc_datasets/110.SemEval/114.restaurant14/Restaurants_Test_Gold.xml.seg.inference',
        'restaurant15': 'integrated_datasets/apc_datasets/110.SemEval/115.restaurant15/restaurant_test.raw.inference',
        'restaurant16': 'integrated_datasets/apc_datasets/110.SemEval/116.restaurant16/restaurant_test.raw.inference',
        'twitter': 'integrated_datasets/apc_datasets/120.Twitter/120.twitter/twitter_test.raw.inference',
        'mams': 'integrated_datasets/apc_datasets/109.MAMS/test.xml.dat.inference',
    }

    pred = []
    gold = []
    texts = open(inference_file[dataset], 'r').readlines()
    for i, text in enumerate(tqdm.tqdm(texts)):
        result = ensemble_predict(apc_classifiers, text, print_result)
        pred.append(result)
        gold.append(text.split('$LABEL$')[-1].strip())
    print(classification_report(gold, pred, digits=4))

if __name__ == '__main__':

    # ensemble_performance('laptop14')
    ensemble_performance('restaurant14')
    # ensemble_performance('restaurant 15')
    ensemble_performance('restaurant16')
    # ensemble_performance('mams')
