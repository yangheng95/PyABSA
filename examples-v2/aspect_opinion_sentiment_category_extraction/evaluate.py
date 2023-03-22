# -*- coding: utf-8 -*-
# file: evaluate.py
# time: 10:54 2023/3/22
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
from itertools import zip_longest

import findfile
import tqdm
from sklearn.metrics import classification_report

from pyabsa import ABSAInstruction, meta_load

if __name__ == "__main__":
    generator = ABSAInstruction.ABSAGenerator(
        # "flant5-base-absa",
        findfile.find_cwd_dir("checkpoint-422"),
    )
    test_files = findfile.find_cwd_files(
        ["integrated_datasets", "acos_datasets", "restaurant", "test"],
        exclude_key=[".ignore", ".txt", ".xlsx"],
    )
    for f in test_files:
        print("Predicting on {}".format(f))
        lines = meta_load(f)
        true_labels = []
        pred_labels = []
        num_total = 0
        acc_count = 0

        for line in tqdm.tqdm(lines):
            true_labels = []
            pred_labels = []
            result = generator.predict(line["text"])
            for true_quadruple in line["labels"]:
                # true_labels.append("{}-{}-{}-{}".format(true_quadruple["aspect"],
                #                                         true_quadruple["polarity"],
                #                                         true_quadruple["opinion"],
                #                                         true_quadruple["category"]))
                # true_labels.append("{}-{}".format(true_quadruple["aspect"], true_quadruple["polarity"]))
                true_labels.append("{}".format(true_quadruple["polarity"]))
            for pred_quadruple in result["Quadruples"]:
                pred_labels.append("{}".format(pred_quadruple["polarity"]))
            for true_label in true_labels:
                num_total += 1
                for pred_label in pred_labels:
                    if true_label.replace(" ", "") == pred_label.replace(" ", ""):
                        acc_count += 1
                    else:
                        print("True: {}, Pred: {}".format(true_label, pred_label))

        print("Accuracy: {}".format(acc_count / num_total))
