# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import os
import pickle

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file


class GloVeRNACDataset(Dataset):

    def __init__(self, tokenizer, config):

        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_dataset_from_file(infer_file, logger=self.config.logger)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, postfix='preparing text classification dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')
                try:
                    exon1, intron, exon2, label = text.strip().split(',')
                except ValueError as e:
                    exon1, intron, exon2 = text.strip().split(',')
                    label = ''
                exon1 = exon1.strip()
                intron = intron.strip()
                exon2 = exon2.strip()
                label = label.strip()
                seq = exon1 + ' ' + intron + ' ' + exon2
                exon1_ids = self.tokenizer.text_to_sequence(exon1)
                intron_ids = self.tokenizer.text_to_sequence(intron)
                exon2_ids = self.tokenizer.text_to_sequence(exon2)

                rna_indices = exon1_ids + intron_ids + exon2_ids

                while len(rna_indices) < self.config.max_seq_len:
                    rna_indices.append(0)

                intron_indices = self.tokenizer.text_to_sequence(intron)
                while len(intron_indices) < self.config.max_seq_len:
                    intron_indices.append(0)

                data = {
                    'ex_id': ex_id,
                    'text_raw': seq,
                    'text_indices': torch.tensor(rna_indices, dtype=torch.long),
                    'intron_indices': torch.tensor(intron_indices, dtype=torch.long),
                    'label': label,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)
    
        return self.data
    

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
