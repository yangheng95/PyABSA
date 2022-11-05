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


class BERTTCDataset(PyABSADataset):
    bert_baseline_input_colses = {
        'mlp': ['text_bert_indices'],
        'bert': ['text_bert_indices'],
    }

    def __init__(self, config, tokenizer, dataset_type='train'):
        self.config = config
        
        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

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
                for x in range(len(seq) // (config.max_seq_len * 2) + 1):
                    _seq = seq[x * (config.max_seq_len * 2):(x + 1) * (config.max_seq_len * 2)]
                    rna_indices = tokenizer.text_to_sequence(_seq, padding=True)

                    data = {
                        'ex_id': torch.tensor(ex_id, dtype=torch.long),
                        'text_bert_indices': torch.tensor(rna_indices, dtype=torch.long),
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
                exon1_ids = tokenizer.text_to_sequence(exon1, padding=False)
                intron_ids = tokenizer.text_to_sequence(intron, padding=False)
                exon2_ids = tokenizer.text_to_sequence(exon2, padding=False)

                rna_indices = exon1_ids + intron_ids + exon2_ids
                while len(rna_indices) < config.max_seq_len:
                    rna_indices.append(tokenizer.tokenizer.pad_token_id)

                data = {
                    'ex_id': torch.tensor(ex_id, dtype=torch.long),
                    'text_bert_indices': torch.tensor(rna_indices, dtype=torch.long),
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
