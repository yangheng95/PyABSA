# -*- coding: utf-8 -*-
# file: ensemble_classification_inference.py
# time: 23/10/2022 15:10
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.


import findfile
import tqdm

from pyabsa import RNAClassification as RNAC
from pyabsa.utils.pyabsa_utils import fprint


def ensemble_predict(rna_classifiers: dict, rna, print_result=False):
    result = []
    for key, rna_classifier in rna_classifiers.items():
        if "bert" in key:
            result += (
                rna_classifier.predict(rna, print_result=print_result)["label"] * 3
            )
        else:
            result += (
                rna_classifier.predict(rna, print_result=print_result)["label"] * 1
            )
    return max(set(result), key=result.count)


if __name__ == "__main__":
    # ckpts = findfile.find_cwd_dirs(or_key=['lstm_degrad', 'bert_mlp_degrad'])
    ckpts = findfile.find_cwd_dirs(or_key=["bert_mlp_degrad"])
    # ckpts = findfile.find_cwd_dirs(or_key=['lstm_degrad'])
    # ckpts = findfile.find_cwd_files('.zip')

    rna_classifiers = {}
    for ckpt in ckpts:
        rna_classifiers[ckpt] = RNAC.RNAClassifier(ckpt)

    # 测试总体准确率
    count = 0
    rnas = open(
        "integrated_datasets/rnac_datasets/degrad/degrad.test.dat.rnac.inference", "r"
    ).readlines()
    for i, rna in enumerate(tqdm.tqdm(rnas)):
        result = ensemble_predict(rna_classifiers, rna, print_result=True)
        if result == rna.split("$LABEL$")[-1].strip():
            count += 1
        fprint(count / (i + 1))

    while True:
        text = input("Please input your RNA sequence: ")
        if text == "exit":
            break
        if text == "":
            continue

        fprint(
            "Predicted Label:",
            ensemble_predict(rna_classifiers, text, print_result=False),
        )
