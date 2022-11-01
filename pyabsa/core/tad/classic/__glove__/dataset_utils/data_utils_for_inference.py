# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import tqdm
from torch.utils.data import Dataset

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets, LABEL_PADDING


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


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


class GloVeTADDataset(Dataset):

    def __init__(self, tokenizer, opt):
        self.glove_input_colses = {
            'tadlstm': ['text_indices']
        }

        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_apc_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, postfix='preparing text classification inference dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                if '!ref!' in text:
                    text, _, labels = text.strip().partition('!ref!')
                    text = text.strip()
                    if labels.count(',') == 2:
                        label, is_adv, adv_train_label = labels.strip().split(',')
                        label, is_adv, adv_train_label = label.strip(), is_adv.strip(), adv_train_label.strip()
                    elif labels.count(',') == 1:
                        label, is_adv = labels.strip().split(',')
                        label, is_adv = label.strip(), is_adv.strip()
                        adv_train_label = '-100'
                    elif labels.count(',') == 0:
                        label = labels.strip()
                        adv_train_label = '-100'
                        is_adv = '-100'
                    else:
                        label = '-100'
                        adv_train_label = '-100'
                        is_adv = '-100'

                    label = int(label)
                    adv_train_label = int(adv_train_label)
                    is_adv = int(is_adv)

                else:
                    text = text.strip()
                    label = -100
                    adv_train_label = -100
                    is_adv = -100

                text_indices = self.tokenizer.text_to_sequence('{}'.format(text))

                data = {
                    'text_indices': text_indices[0],

                    'text_raw': text,

                    'label': label,

                    'adv_train_label': adv_train_label,

                    'is_adv': is_adv,

                    # 'label': self.opt.label_to_index.get(label, -100) if isinstance(label, str) else label,
                    #
                    # 'adv_train_label': self.opt.adv_train_label_to_index.get(adv_train_label, -100) if isinstance(adv_train_label, str) else adv_train_label,
                    #
                    # 'is_adv': self.opt.is_adv_to_index.get(is_adv, -100) if isinstance(is_adv, str) else is_adv,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text)
                else:
                    raise e

        self.all_data = all_data
        return self.all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)
