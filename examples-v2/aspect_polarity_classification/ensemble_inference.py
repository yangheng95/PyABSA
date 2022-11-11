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

from pyabsa import AspectPolarityClassification as APC


def ensemble_predict(apc_classifiers: dict, text, print_result=False):
    result = []
    for key, apc_classifier in apc_classifiers.items():
        result += apc_classifier.predict(text, print_result=print_result)['sentiment']
    return max(set(result), key=result.count)

if __name__ == '__main__':
    # # ckpts = findfile.find_cwd_dirs('fast_lsa_t_v2_Laptop14_acc')
    # # ckpts = findfile.find_cwd_dirs('fast_lsa_s_v2_Laptop14_acc')
    # ckpts = findfile.find_cwd_dirs('Laptop14_acc')
    # random.shuffle(ckpts)
    # apc_classifiers = {}
    # for ckpt in ckpts[:5]:
    #     apc_classifiers[ckpt] = (APC.SentimentClassifier(ckpt))
    #
    # # 测试总体准确率
    # count = 0
    # texts = open('integrated_datasets/apc_datasets/110.SemEval/113.laptop14/Laptops_Test_Gold.xml.seg.inference', 'r').readlines()
    # for i, text in enumerate(texts):
    #
    #     result = ensemble_predict(apc_classifiers, text, print_result=False)
    #     if result == text.split('$LABEL$')[-1].strip():
    #         count += 1
    #     print(count / (i+1))


    # ckpts = findfile.find_cwd_dirs('fast_lsa_t_v2_Restaurant14_acc')
    # ckpts = findfile.find_cwd_dirs('fast_lsa_s_v2_Restaurant14_acc')
    ckpts = findfile.find_cwd_dirs('Restaurant14_acc')
    random.shuffle(ckpts)
    apc_classifiers = {}
    for ckpt in ckpts[:5]:
        apc_classifiers[ckpt] = (APC.SentimentClassifier(ckpt))

    # 测试总体准确率
    count = 0
    texts = open('integrated_datasets/apc_datasets/110.SemEval/114.restaurant14/Restaurants_Test_Gold.xml.seg.inference', 'r').readlines()
    for i, text in enumerate(texts):

        result = ensemble_predict(apc_classifiers, text, print_result=False)
        if result == text.split('$LABEL$')[-1].strip():
            count += 1
        print(count / (i+1))
