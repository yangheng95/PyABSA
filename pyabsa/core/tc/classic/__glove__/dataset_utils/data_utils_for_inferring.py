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


class GloVeClassificationDataset(Dataset):

    def __init__(self, tokenizer, opt):
        self.glove_input_colses = {
            'lstm': ['text_indices']
        }

        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str):
        self.process_data(self.parse_sample(text))

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_apc_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []

        if len(samples) > 1:
            it = tqdm.tqdm(samples, postfix='building word indices...')
        else:
            it = samples

        for text in it:
            try:
                # handle for empty lines in inferring_tutorials dataset_utils
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                if '!ref!' in text:
                    text, label = text.split('!ref!')[0].strip(), text.split('!ref!')[1].strip()
                    label = int(label) if label else LABEL_PADDING
                    text = text.replace('[PADDING]', '')

                    if label < 0:
                        raise RuntimeError(
                            'Invalid label detected, only please label the sentiment between {0, N-1} '
                            '(assume there are N types of labels.)')
                else:
                    label = LABEL_PADDING

                text_indices = self.tokenizer.text_to_sequence(text)

                data = {
                    'text_indices': text_indices
                    if 'text_indices' in self.opt.model.inputs else 0,

                    'text_raw': text,

                    'label': label,
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
