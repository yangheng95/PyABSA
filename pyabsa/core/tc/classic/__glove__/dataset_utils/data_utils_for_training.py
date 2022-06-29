# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import tqdm
from torch.utils.data import Dataset

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, pad_and_truncate


def build_tokenizer(dataset_list, max_seq_len, dat_fname, opt):
    dataset_name = os.path.basename(opt.dataset_name)
    if not os.path.exists('run/{}'.format(dataset_name)):
        os.makedirs('run/{}'.format(dataset_name))
    tokenizer_path = 'run/{}/{}'.format(dataset_name, dat_fname)
    if os.path.exists(tokenizer_path):
        print('Loading tokenizer on {}'.format(tokenizer_path))
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    else:
        text = ''
        for dataset_type in dataset_list:
            for file in dataset_list[dataset_type]:
                fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                lines = fin.readlines()
                fin.close()
                for i in range(0, len(lines)):
                    text += lines[i].lower().strip()

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
    return tokenizer


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class GloVeTCDataset(Dataset):
    glove_input_colses = {
        'lstm': ['text_indices']
    }

    def __init__(self, dataset_list, tokenizer, opt):
        lines = load_apc_datasets(dataset_list)

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split('$LABEL$')
            text, label = line[0], line[1]
            text = text.strip().lower()
            label = label.strip().lower()
            text_indices = tokenizer.text_to_sequence(text)

            label = int(label)

            data = {
                'text_indices': text_indices,

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
