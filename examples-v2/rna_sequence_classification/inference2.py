# -*- coding: utf-8 -*-
# file: inference.py
# time: 23/10/2022 15:10
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.


import findfile
import tqdm

from pyabsa.tasks import RNABiClassification as RNAC
from pyabsa.utils.pyabsa_utils import fprint


def ensemble_predict(rna_classifiers: dict, rna, print_result=False):
    results1 = []
    results2 = []
    for key, rna_classifier in rna_classifiers.items():
        res = rna_classifier.predict(rna, ignore_error=False, print_result=print_result)
        if 'bert' in key:
            for i in range(3):
                results1.append(res['decay_label'])
                results2.append(res['seq_label'])
        else:
            results1.append(res['decay_label'])
            results2.append(res['seq_label'])
    return max(set(results1), key=results1.count), max(set(results2), key=results2.count)


if __name__ == '__main__':

    # ckpts = findfile.find_cwd_dirs(or_key=['lstm_degrad-v2', 'bert_mlp_degrad-v2'])
    # ckpts = findfile.find_cwd_dirs(or_key=['bert_mlp_degrad-v2'])
    # ckpts = findfile.find_cwd_dirs(or_key=['bert_mlp_degrad-v2'])
    ckpts = findfile.find_cwd_dirs(or_key=['lstm_degrad-v2'])
    # ckpts = findfile.find_cwd_files('.zip')

    rna_classifiers = {}
    for ckpt in ckpts[:]:
        rna_classifiers[ckpt] = (RNAC.RNAClassifier(ckpt))

    # 测试总体准确率
    count1 = 0
    count2 = 0
    rnas = open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.test.dat.rnac.inference', 'r').readlines()
    for i, rna in enumerate(tqdm.tqdm(rnas)):
        results = ensemble_predict(rna_classifiers, rna, print_result=True)
        if results[0] == rna.split('$LABEL$')[-1].strip().split(',')[0]:
            count1 += 1
        if results[1] == rna.split('$LABEL$')[-1].strip().split(',')[1]:
            count2 += 1
        fprint('Decay classification accuracy:', count1 / (i + 1))
        fprint('RNA classification accuracy:', count2 / (i + 1))

    rnas = [
        'ATGGGATAATGGTTTCGTACCAAAAGCTGGTGCGTTCCTTCCTTTTGGTGCTGGAAGCCATCTATGCCCGGGAAATGATCTGGCTAAGCTCGAGATTTCAATTTTTCTTCATCATTTCCTCCTCAAATATCAGGTGAAACGGAGCAACCCCGAATGTCCAGTGATGTATCTGCCTCATACCAGACCAACTGATAATTGCT$LABEL$1,cds',
        'TGTGAGTGAAGAAGATAATGCAGACTCACCTTTTGGTGGGACCTATCCCACTCAAAGGCTACCGTCGATTCTCTTCCTCCTCCTTCTCCGGCGATCTCCTCCCTCCGTCGTCTAACCCTATCGGCCGAGACCTATTCCCTCACCGTCGAAGGCACCGCGACGGCAAATCTCGGAGTTACCGTAATCGCTCGAAAACGACG$LABEL$1,cds',
        'TAAACCGTATTTAAATGGACGATCGATGTATCTTTTGAACAGTTTCCTCGTGAATGCGTTAGGTATGATGGGTTCCGGGAAAACGACTGTAGGGAAGATTATGGCAAGATCGCTTGGTTATACATTCTTTGATTGTGACACTTTGATCGAGCAGGCTATGAAGGGAACTTCTGTAGCTGAGATATTTGAGCATTTCGGTG$LABEL$1,cds',
        'AACCTCAAACCAGAAACACAAGCAACTCTTGTGGACAATATAATGGCCCTAGGATCTGAATGGTTTCAGTCACCCTTGAAGCTTACGACTTTGATTTCTATCTACAAAGTCTTTATTGCACGTAGATACGCCCTCCAGGTGATAAAGGACGTTTTCACGAGGAGGAAAGCGTCCAGAGAAATGTGCGGAGACTTCCTCGA$LABEL$1,cds',
        'CCGTTTGAGTGGAGACGAAGGCGTTTCCGGTTCTCTTCTCTCGTCGGAGTTCTGAGGTAAAAAAAGAATAAGGAGAAGAAGAAAAGCAAAAGCATAAAAGAGAGTAGCAAAGACTGAGAATGGAAAGCTTGGACACTAATTTTCCTGTGCGCCATAGAAAGGTCTCGTTTGAAAGTAAGGGAAACAAGACAGAGATTGTG$LABEL$1,5utr',

    ]
    # classifier = RNAC.RNAClassifier('lstm_degrad_acc_83.03_f1_82.25')
    for rna in rnas:
        _, _, labels = rna.partition('$LABEL$')
        labels = labels.split(',')
        decay_label, seq_label = ensemble_predict(rna_classifiers, rna, print_result=False)
        fprint('Predicted Label:', decay_label, decay_label == labels[0], seq_label, seq_label == labels[1])

    while True:
        rna = input('Please input your RNA sequence: ')
        if rna == 'exit':
            break
        if rna == '':
            continue
        _, _, labels = rna.partition('$LABEL$')
        labels = labels.split(',')
        try:
            decay_label, seq_label = ensemble_predict(rna_classifiers, rna, print_result=False)
            fprint('Predicted Label:', decay_label, decay_label == labels[0], seq_label, seq_label == labels[1])
        except Exception as e:
            fprint(e)
