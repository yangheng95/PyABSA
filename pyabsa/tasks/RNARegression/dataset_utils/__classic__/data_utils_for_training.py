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


class GloVeRNARDataset(PyABSADataset):
    glove_input_colses = {
        'lstm': ['text_indices'],
        'cnn': ['text_indices'],
        'transformer': ['text_indices'],
        'mhsa': ['text_indices'],
    }

    def __init__(self, config, tokenizer, dataset_type='train'):
        self.config = config

        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

        label_set = set()

        for ex_id, i in enumerate(tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...')):
            line = lines[i].strip().split('\t') if '\t' in lines[i] else lines[i].strip().split(',')
            try:
                _, label, r1r2_label, r1r3_label, r2r3_label, seq = line[0], line[1], line[2], line[3], line[4], line[5]
                label = float(label.strip())

                # r1r2_label = float(r1r2_label.strip())
                # r1r3_label = float(r1r3_label.strip())
                # r2r3_label = float(r2r3_label.strip())
                # if len(seq) > 2 * config.max_seq_len:
                #     continue
                # for x in range(len(seq) // (config.max_seq_len * 2) + 1):
                #     _seq = seq[x * (config.max_seq_len * 2):(x + 1) * (config.max_seq_len * 2)]
                for x in range(len(seq) // (config.max_seq_len * 3) + 1):
                    _seq = seq[x * (config.max_seq_len * 3):(x + 1) * (config.max_seq_len * 3)]
                    rna_indices = tokenizer.text_to_sequence(_seq)
                    while len(rna_indices) < config.max_seq_len:
                        rna_indices.append(0)
                    if any(rna_indices):
                        data = {
                            'ex_id': torch.tensor(ex_id, dtype=torch.long),
                            'text_indices': torch.tensor(rna_indices, dtype=torch.long),
                            'label': torch.tensor(label, dtype=torch.float32),
                            # 'r1r2_label': torch.tensor(r1r2_label, dtype=torch.float32),
                            # 'r1r3_label': torch.tensor(r1r3_label, dtype=torch.float32),
                            # 'r2r3_label': torch.tensor(r2r3_label, dtype=torch.float32),
                        }

                        all_data.append(data)

            except Exception as e:
                exon1, intron, exon2, label = line[0], line[1], line[2], line[3]
                label = float(label.strip())
                seq = exon1 + intron + exon2
                exon1_ids = tokenizer.text_to_sequence(exon1)
                intron_ids = tokenizer.text_to_sequence(intron)
                exon2_ids = tokenizer.text_to_sequence(exon2)

                rna_indices = exon1_ids + intron_ids + exon2_ids
                while len(rna_indices) < config.max_seq_len:
                    rna_indices.append(0)
                while len(intron_ids) < config.max_seq_len:
                    intron_ids.append(0)

                data = {
                    'ex_id': torch.tensor(ex_id, dtype=torch.long),
                    'text_indices': torch.tensor(rna_indices, dtype=torch.long),
                    'label': torch.tensor(label, dtype=torch.float32)
                }

                all_data.append(data)

        config.output_dim = 1
        self.data = all_data

        super().__init__(config)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
