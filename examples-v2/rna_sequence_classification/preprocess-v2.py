# -*- coding: utf-8 -*-
# file: preprocess.py
# time: 09/11/2022 22:33
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import random

from pyabsa.utils.pyabsa_utils import fprint


def preprocess_rna():
    def load_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines

    train_data = load_file('integrated_datasets/rnac_datasets/degrad/degrad.train.dat.rnac')
    test_data = load_file('integrated_datasets/rnac_datasets/degrad/degrad.test.dat.rnac')
    valid_data = load_file('integrated_datasets/rnac_datasets/degrad/degrad.valid.dat.rnac')

    positive_data = load_file('integrated_datasets/rnac_datasets/degrad-v2/Degrad_XRN4_DL_1.tsv')
    negative_data = load_file('integrated_datasets/rnac_datasets/degrad-v2/Degrad_XRN4_DL_0.tsv')
    positive_rna_label = dict()
    negative_rna_label = dict()
    for line in positive_data:
        if line.split('\t')[0].strip() not in positive_rna_label:
            positive_rna_label[line.split('\t')[-1].strip()] = [line.split('\t')[2].strip()]
        else:
            positive_rna_label[line.split('\t')[-1].strip()].append(line.split('\t')[2].strip())

    for line in negative_data:
        if line.split('\t')[0].strip() not in negative_rna_label:
            negative_rna_label[line.split('\t')[-1].strip()] = [line.split('\t')[2].strip()]
        else:
            negative_rna_label[line.split('\t')[-1].strip()].append(line.split('\t')[2].strip())

    for line in positive_data:
        if line.split('\t')[0].strip() not in positive_rna_label:
            positive_rna_label[line.split('\t')[-1].strip()] = [line.split('\t')[2].strip()]
        else:
            positive_rna_label[line.split('\t')[-1].strip()].append(line.split('\t')[2].strip())

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.train.dat.rnac', 'w') as f:
        for line in train_data:
            line = line.strip()
            if line.split('$LABEL$')[0].strip() in positive_rna_label:
                line = line.strip() + ',' + positive_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            else:
                line = line.strip() + ',' + negative_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            f.write(line)

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.test.dat.rnac', 'w') as f:
        for line in test_data:
            if line.split('$LABEL$')[0].strip() in positive_rna_label:
                line = line.strip() + ',' + positive_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            else:
                line = line.strip() + ',' + negative_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            f.write(line)

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.valid.dat.rnac', 'w') as f:
        for line in valid_data:
            if line.split('$LABEL$')[0].strip() in positive_rna_label:
                line = line.strip() + ',' + positive_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            else:
                line = line.strip() + ',' + negative_rna_label[line.split('$LABEL$')[0].strip()][0] + '\n'
            f.write(line)

    random.shuffle(negative_data)
    random.shuffle(positive_data)
    negative_data = negative_data[: len(negative_data) // 100]

    positive_rna_name_list = dict()
    negative_rna_name_list = dict()
    for line in positive_data:
        if line.split('\t')[0].strip() not in positive_rna_name_list:
            positive_rna_name_list[line.split('\t')[0].strip()] = [(line.split('\t')[-1].strip(), line.split('\t')[2].strip(), '1')]
        else:
            positive_rna_name_list[line.split('\t')[0].strip()].append((line.split('\t')[-1].strip(), line.split('\t')[2].strip(), '1'))

    for line in negative_data:
        if line.split('\t')[0].strip() not in negative_rna_name_list:
            negative_rna_name_list[line.split('\t')[0].strip()] = [(line.split('\t')[-1].strip(), line.split('\t')[2].strip(), '0')]
        else:
            negative_rna_name_list[line.split('\t')[0].strip()].append((line.split('\t')[-1].strip(), line.split('\t')[2].strip(), '0'))

    positive_train_names = list(positive_rna_name_list.keys())[:int(len(positive_rna_name_list) * 0.8)]
    positive_test_names = list(positive_rna_name_list.keys())[int(len(positive_rna_name_list) * 0.8):int(len(positive_rna_name_list) * 0.9)]
    positive_valid_names = list(positive_rna_name_list.keys())[int(len(positive_rna_name_list) * 0.9):]

    negative_train_names = list(negative_rna_name_list.keys())[:int(len(negative_rna_name_list) * 0.8)]
    negative_test_names = list(negative_rna_name_list.keys())[int(len(negative_rna_name_list) * 0.8):int(len(negative_rna_name_list) * 0.9)]
    negative_valid_names = list(negative_rna_name_list.keys())[int(len(negative_rna_name_list) * 0.9):]

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.train.dat.rnac', 'w') as f:
        for data in positive_train_names + negative_train_names:
            if data in positive_train_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

            if data in negative_train_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.test.dat.rnac', 'w') as f:
        for data in positive_test_names + negative_test_names:
            if data in positive_test_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

            if data in negative_test_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.valid.dat.rnac', 'w') as f:
        for data in positive_valid_names + negative_valid_names:
            if data in positive_valid_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

            if data in negative_valid_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

    with open('integrated_datasets/rnac_datasets/degrad-v2/degrad-v2.test.dat.rnac.inference', 'w') as f:
        for data in positive_test_names + negative_test_names:
            if data in positive_test_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')

            if data in negative_test_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq[0] + '$LABEL$' + seq[1] + ',' + seq[2] + '\n')


if __name__ == '__main__':
    preprocess_rna()
    fprint('Done!')
