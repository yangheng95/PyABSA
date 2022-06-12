# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import tqdm
from findfile import find_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, TransformerConnectionError


class Tokenizer4Pretraining:
    def __init__(self, max_seq_len, opt):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert, do_lower_case='uncased' in opt.pretrained_bert)
        except ValueError as e:
            raise TransformerConnectionError()

        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return self.tokenizer.encode(text, truncation=True, padding='max_length', max_length=self.max_seq_len, return_tensors='pt')


class BERTTCDataset(Dataset):
    bert_baseline_input_colses = {
        'bert': ['text_bert_indices'],
        'huggingfaceencoder': ['text_bert_indices'],
    }

    def __init__(self, dataset_list, tokenizer, opt):
        lines = load_apc_datasets(dataset_list)

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split('$LABEL$')
            text, label = line[0], line[1]
            text = text.strip()
            label = label.strip()
            text_indices = tokenizer.text_to_sequence('{} {} {}'.format(tokenizer.tokenizer.cls_token, text, tokenizer.tokenizer.sep_token))

            data = {
                'text_bert_indices': text_indices[0],
                'label': label,
            }

            label_set.add(label)

            all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, opt)
        opt.polarities_dim = len(label_set)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
