# -*- coding: utf-8 -*-
# file: train_rna_classification.py
# time: 22/10/2022 16:36
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.
import random

from pyabsa import RNAClassification as RNAC
from pyabsa.utils.data_utils.dataset_item import DatasetItem
from pyabsa.utils.pyabsa_utils import fprint


def preprocess_rna():
    fprint('splitting data...')

    def load_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines

    positive_data = load_file('integrated_datasets/rnac_datasets/degrad/Degrad_XRN4_DL_1.tsv')
    # negative_data = load_file('integrated_datasets/rnac_datasets/degrad/Degrad_XRN4_DL_0.tsv')
    negative_data = load_file('integrated_datasets/rnac_datasets/degrad/Degrad_XRN4_DL_0_sample.tsv')
    random.shuffle(negative_data)
    negative_data = negative_data[: len(negative_data)]

    positive_rna_name_list = dict()
    negative_rna_name_list = dict()
    for line in positive_data:
        if line.split('\t')[0].strip() not in positive_rna_name_list:
            positive_rna_name_list[line.split('\t')[0].strip()] = [line.split('\t')[-1].strip()]
        else:
            positive_rna_name_list[line.split('\t')[0].strip()].append(line.split('\t')[-1].strip())

    for line in negative_data:
        if line.split('\t')[0].strip() not in negative_rna_name_list:
            negative_rna_name_list[line.split('\t')[0].strip()] = [line.split('\t')[-1].strip()]
        else:
            negative_rna_name_list[line.split('\t')[0].strip()].append(line.split('\t')[-1].strip())

    positive_train_names = list(positive_rna_name_list.keys())[:int(len(positive_rna_name_list) * 0.8)]
    positive_test_names = list(positive_rna_name_list.keys())[
                          int(len(positive_rna_name_list) * 0.8):int(len(positive_rna_name_list) * 0.9)]
    positive_valid_names = list(positive_rna_name_list.keys())[int(len(positive_rna_name_list) * 0.9):]

    negative_train_names = list(negative_rna_name_list.keys())[:int(len(negative_rna_name_list) * 0.8)]
    negative_test_names = list(negative_rna_name_list.keys())[
                          int(len(negative_rna_name_list) * 0.8):int(len(negative_rna_name_list) * 0.9)]
    negative_valid_names = list(negative_rna_name_list.keys())[int(len(negative_rna_name_list) * 0.9):]

    with open('integrated_datasets/rnac_datasets/degrad/degrad.train.dat.rnac', 'w') as f:
        for data in positive_train_names + negative_train_names:
            if data in positive_train_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq + '$LABEL$1\n')

            if data in negative_train_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq + '$LABEL$0\n')

    with open('integrated_datasets/rnac_datasets/degrad/degrad.test.dat.rnac', 'w') as f:
        for data in positive_test_names + negative_test_names:
            if data in positive_test_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq + '$LABEL$1\n')

            if data in negative_test_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq + '$LABEL$0\n')

    with open('integrated_datasets/rnac_datasets/degrad/degrad.valid.dat.rnac', 'w') as f:
        for data in positive_valid_names + negative_valid_names:
            if data in positive_valid_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq + '$LABEL$1\n')

            if data in negative_valid_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq + '$LABEL$0\n')

    with open('integrated_datasets/rnac_datasets/degrad/degrad.test.dat.rnac.inference', 'w') as f:
        for data in positive_test_names + negative_test_names:
            if data in positive_test_names:
                for seq in positive_rna_name_list[data]:
                    f.write(seq + '$LABEL$1\n')

            if data in negative_test_names:
                for seq in negative_rna_name_list[data]:
                    f.write(seq + '$LABEL$0\n')


for i in range(10):
    # preprocess_rna()

    config = RNAC.RNACConfigManager.get_rnac_config_glove()
    config.model = RNAC.GloVeRNACModelList.LSTM
    config.num_epoch = 10
    config.pretrained_bert = 'rna_bpe_tokenizer'
    config.evaluate_begin = 0
    config.max_seq_len = 100
    # config.hidden_dim = 768
    # config.embed_dim = 768
    config.cache_dataset = False
    # config.cache_dataset = True
    config.dropout = 0.5
    config.num_lstm_layer = 1
    config.do_lower_case = False
    config.seed = [random.randint(0, 10000) for _ in range(1)]
    config.log_step = -1
    config.show_metric = True
    config.l2reg = 0.001
    config.save_last_ckpt_only = True
    config.num_mhsa_layer = 1

    dataset = DatasetItem('degrad')

    classifier = RNAC.RNACTrainer(config=config,
                                  dataset=dataset,
                                  checkpoint_save_mode=1,
                                  auto_device=True
                                  ).load_trained_model()

rnas = [
    'GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGGAGCGGTGGGGTATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC',
    'GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGAGGCGGTGGGGTATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC',
    'GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGAAGCAGTAGGATATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC',
    'GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGAATTTGAAGCAGTAGAATATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC',
    'TTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGAATTTGAAGCAGTAGAATATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC'
]
# classifier = RNAC.RNAClassifier('lstm_degrad_acc_83.03_f1_82.25')
for rna in rnas:
    classifier.predict(rna)

# classifier.batch_predict(dataset)
