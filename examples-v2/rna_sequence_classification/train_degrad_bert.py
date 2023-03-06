# -*- coding: utf-8 -*-
# file: train_degrad.py
# time: 06/11/2022 01:43
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os
import pickle

# def load_file(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#         return lines
#
#
# if os.path.exists('integrated_datasets/rnac_datasets/degrad/degrad.pkl'):
#     dataset_dict = pickle.load(open('integrated_datasets/rnac_datasets/degrad/degrad.pkl', 'rb'))
#
# else:
#     positive_data = load_file('integrated_datasets/rnac_datasets/degrad/Degrad_XRN4_DL_1.tsv')
#     negative_data = load_file('integrated_datasets/rnac_datasets/degrad/Degrad_XRN4_DL_0_sample.tsv')
#
#     positive_rna_name_list = set()
#     negative_rna_name_list = set()
#     for line in positive_data:
#         positive_rna_name_list.add(line.split('\t')[0].strip())
#
#     for line in negative_data:
#         negative_rna_name_list.add(line.split('\t')[0].strip())
#
#     positive_train_names = list(positive_rna_name_list)[:int(len(positive_rna_name_list) * 0.8)]
#     positive_test_names = list(positive_rna_name_list)[int(len(positive_rna_name_list) * 0.8):int(len(positive_rna_name_list) * 0.9)]
#     positive_valid_names = list(positive_rna_name_list)[int(len(positive_rna_name_list) * 0.9):]
#
#     negative_train_names = list(negative_rna_name_list)[:int(len(negative_rna_name_list) * 0.8)]
#     negative_test_names = list(negative_rna_name_list)[int(len(negative_rna_name_list) * 0.8):int(len(negative_rna_name_list) * 0.9)]
#     negative_valid_names = list(negative_rna_name_list)[int(len(negative_rna_name_list) * 0.9):]
#
#     from pyabsa import DatasetDict
#
#     dataset_dict = DatasetDict()
#
#     for train_name in positive_train_names + negative_train_names:
#         for line in positive_data:
#             if line.split('\t')[0].strip() == train_name:
#                 dataset_dict['train'].append({'data': line.split('\t')[-1], 'label': '1'})
#         for line in negative_data:
#             if line.split('\t')[0].strip() == train_name:
#                 dataset_dict['train'].append({'data': line.split('\t')[-1], 'label': '0'})
#
#     for test_name in positive_test_names + negative_test_names:
#         for line in positive_data:
#             if line.split('\t')[0].strip() == test_name:
#                 dataset_dict['test'].append({'data': line.split('\t')[-1], 'label': '1'})
#         for line in negative_data:
#             if line.split('\t')[0].strip() == test_name:
#                 dataset_dict['test'].append({'data': line.split('\t')[-1], 'label': '0'})
#
#     for valid_name in positive_valid_names + negative_valid_names:
#         for line in positive_data:
#             if line.split('\t')[0].strip() == valid_name:
#                 dataset_dict['valid'].append({'data': line.split('\t')[-1], 'label': '1'})
#         for line in negative_data:
#             if line.split('\t')[0].strip() == valid_name:
#                 dataset_dict['valid'].append({'data': line.split('\t')[-1], 'label': '0'})
#
#
# with open('integrated_datasets/rnac_datasets/degrad/degrad.train.dat.rnac', 'w') as f:
#     for data in dataset_dict['train']:
#         f.write(data['data'].strip() + '$LABEL$' + data['label'] + '\n')
#
# with open('integrated_datasets/rnac_datasets/degrad/degrad.test.dat.rnac', 'w') as f:
#     for data in dataset_dict['test']:
#         f.write(data['data'].strip() + '$LABEL$' + data['label'] + '\n')
#
# with open('integrated_datasets/rnac_datasets/degrad/degrad.valid.dat.rnac', 'w') as f:
#     for data in dataset_dict['valid']:
#         f.write(data['data'].strip() + '$LABEL$' + data['label'] + '\n')

import random

from pyabsa import RNAClassification as RNAC, DatasetItem
from pyabsa.utils.pyabsa_utils import fprint

# preprocess_rna()
config = RNAC.RNACConfigManager.get_rnac_config_english()
config.model = RNAC.BERTRNACModelList.BERT_MLP
config.num_epoch = 5
config.pretrained_bert = "roberta-base"
config.evaluate_begin = 0
config.max_seq_len = 200
config.hidden_dim = 768
config.embed_dim = 768
# config.cache_dataset = False
config.cache_dataset = False
config.dropout = 0.5
config.l2reg = 0
config.batch_size = 16
config.learning_rate = 1e-5
config.show_metric = True
config.num_lstm_layer = 1
config.do_lower_case = False
config.seed = [random.randint(0, 10000) for _ in range(3)]
config.log_step = -1
config.save_last_ckpt_only = True
config.num_mhsa_layer = 1

dataset = DatasetItem("degrad")

classifier = RNAC.RNACTrainer(
    config=config, dataset=dataset, checkpoint_save_mode=1, auto_device=True
).load_trained_model()

# classifier = RNAC.RNAClassifier('bert_mlp_degrad_acc_87.44_f1_86.99', auto_device=True)
# classifier = RNAC.RNAClassifier('bert_mlp_degrad_acc_83.41_f1_82.66', auto_device=True)

rnas = [
    "GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGGAGCGGTGGGGTATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC",
    "GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGAGGCGGTGGGGTATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC",
    "GTGCGATCGTTGATCTTGTGGCTTGTGAGCCGTCGGATTCCACGGAGAGGCGAGAGACAGCGAGGAAGTGGTCGAGGAGGATGAGGAATAGTGGGTTTGAAGCAGTAGGATATAGTGATGAGGTGGCGGATGATGTCAGAGCTTTGTTGAGGAGATATAAAGAAGGTGTTTGGTCGATGGTACAGTGTCCTGATGCCGCCGGAATATTCC",
    "GTGCGATCGTTGATCTTGTGACTTGTGAGCCGTCGAATTCCACGAAGAGACGAGAGACAGCGAGAAAGTGATCGAGAAGAATGAGAAATAGTGAATTTGAAGCAGTAGAATATAGTGATGAGATGACGAATGATGTCAGAGCTTTGTTGAGAAGATATAAAGAAGATGTTTGATCGATGATACAGTGTCCTGATGCCGCCGAAATATTCC",
]
for rna in rnas:
    fprint(classifier.predict(rna + "$LABEL$"))

classifier.batch_predict(dataset)
