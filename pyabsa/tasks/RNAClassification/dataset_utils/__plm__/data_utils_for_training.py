# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import torch
import tqdm
from findfile import find_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels



class BERTTCDataset(PyABSADataset):
    bert_baseline_input_colses = {
        'bert': ['text_bert_indices'],
        'bert_mlp': ['text_bert_indices'],

    }

    def __init__(self, config, tokenizer, dataset_type='train'):

        self.config = config
        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split(',')
            exon1, intron, exon2, label = line[0], line[1], line[2], line[3]
            exon1 = exon1.strip()
            intron = intron.strip()
            exon2 = exon2.strip()
            label = label.strip()
            exon1_ids = tokenizer.text_to_sequence(exon1, padding='do_not_pad')
            intron_ids = tokenizer.text_to_sequence(intron, padding='do_not_pad')
            exon2_ids = tokenizer.text_to_sequence(exon2, padding='do_not_pad')
            rna_indices = [tokenizer.tokenizer.cls_token_id] + exon1_ids + intron_ids + exon2_ids + [tokenizer.tokenizer.sep_token_id]

            while len(rna_indices) < self.config.max_seq_len:
                rna_indices.append(tokenizer.tokenizer.pad_token_id)

            intron_indices = tokenizer.text_to_sequence(intron)

            data = {
                'text_bert_indices': torch.tensor(rna_indices, dtype=torch.long),
                'intron_indices': torch.tensor(intron_indices, dtype=torch.long),
                'label': label,
            }

            label_set.add(label)

            all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)

        self.data = all_data

        super().__init__(config)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
