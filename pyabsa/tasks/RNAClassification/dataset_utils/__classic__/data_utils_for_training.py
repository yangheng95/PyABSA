# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import torch
import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels

class GloVeRNACDataset(PyABSADataset):
    glove_input_colses = {
        'lstm': ['text_indices', 'intron_indices'],
        'cnn': ['text_indices', 'intron_indices'],
        'transformer': ['text_indices', 'intron_indices'],
        'mhsa': ['text_indices', 'intron_indices'],
    }

    def __init__(self, config, tokenizer, dataset_type='train'):

        self.config = config
        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

        label_set = set()

        for ex_id, i in enumerate(tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...')):
            line = lines[i].strip().split(',')
            exon1, intron, exon2, label = line[0], line[1], line[2], line[3]
            exon1 = exon1.strip()
            intron = intron.strip()
            exon2 = exon2.strip()
            label = label.strip()
            exon1_ids = tokenizer.text_to_sequence(exon1, padding_len=None)
            intron_ids = tokenizer.text_to_sequence(intron, padding_len=None)
            exon2_ids = tokenizer.text_to_sequence(exon2, padding_len=None)

            rna_indices = exon1_ids + intron_ids + exon2_ids

            while len(rna_indices) < self.config.max_seq_len:
                rna_indices.append(0)

            intron_indices = tokenizer.text_to_sequence(intron)
            while len(intron_indices) < self.config.max_seq_len:
                intron_indices.append(0)

            data = {
                'ex_id': ex_id,
                'text_indices': torch.tensor(rna_indices, dtype=torch.long),
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
