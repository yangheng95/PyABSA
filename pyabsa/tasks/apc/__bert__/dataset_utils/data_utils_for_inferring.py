# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import tqdm

from torch.utils.data import Dataset

from pyabsa.tasks.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.tasks.apc.__glove__.dataset_utils.dependency_graph import (prepare_dependency_graph,
                                                                       dependency_adj_matrix)

from pyabsa.tasks.apc.dataset_utils.apc_utils import SENTIMENT_PADDING
from transformers import AutoTokenizer


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


class Tokenizer4Pretraining:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class BERTBaselineABSADataset(Dataset):

    def __init__(self, tokenizer, opt):
        self.bert_baseline_input_colses = {
            'lstm_bert': ['text_indices'],
            'td_lstm_bert': ['left_with_aspect_indices', 'right_with_aspect_indices'],
            'tc_lstm_bert': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
            'atae_lstm_bert': ['text_indices', 'aspect_indices'],
            'ian_bert': ['text_indices', 'aspect_indices'],
            'memnet_bert': ['context_indices', 'aspect_indices'],
            'ram_bert': ['text_indices', 'aspect_indices', 'left_indices'],
            'cabasc_bert': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
            'tnet_lf_bert': ['text_indices', 'aspect_indices', 'aspect_boundary'],
            'aoa_bert': ['text_indices', 'aspect_indices'],
            'mgan_bert': ['text_indices', 'aspect_indices', 'left_indices'],
            'asgcn_bert': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        }

        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        _text = text
        samples = []
        try:
            if '!sent!' not in text:
                splits = text.split('[ASP]')
                for i in range(0, len(splits) - 1, 2):
                    sample = text.replace('[ASP]', '').replace(splits[i + 1], '[ASP]' + splits[i + 1] + '[ASP]')
                    samples.append(sample)
            else:
                text, ref_sent = text.split('!sent!')
                ref_sent = ref_sent.split()
                text = '[PADDING] ' + text + ' [PADDING]'
                splits = text.split('[ASP]')

                if int((len(splits) - 1) / 2) == len(ref_sent):
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        sample += ' !sent! ' + str(ref_sent[int(i / 2)])
                        samples.append(sample.replace('[TEMP]', '[ASP]'))
                else:
                    print(_text,
                          ' -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.')
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        samples.append(sample.replace('[TEMP]', '[ASP]'))

        except:
            print('Invalid Input:', _text)
        return samples

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

        for text in tqdm.tqdm(samples, postfix='building word indices...'):
            try:
                # handle for empty lines in inferring_tutorials dataset
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '!sent!' in text:
                    text, polarity = text.split('!sent!')[0].strip(), text.split('!sent!')[1].strip()
                    polarity = int(polarity) if polarity else SENTIMENT_PADDING
                    text = text.replace('[PADDING]', '')

                    if polarity < 0:
                        raise RuntimeError(
                            'Invalid sentiment label detected, only please label the sentiment between {0, N-1} '
                            '(assume there are N types of sentiment polarities.)')
                else:
                    polarity = SENTIMENT_PADDING

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split('[ASP]')
                text_left = text_left.replace('[PADDING] ', '')
                text_right = text_right.replace(' [PADDING]', '')
                text_indices = self.tokenizer.text_to_sequence('[CLS] ' + text_left + ' ' + aspect + ' ' + text_right + " [SEP]")
                context_indices = self.tokenizer.text_to_sequence(text_left + text_right)
                left_indices = self.tokenizer.text_to_sequence(text_left)
                left_with_aspect_indices = self.tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " [SEP]")
                right_indices = self.tokenizer.text_to_sequence(text_right, reverse=False)
                right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=False)
                aspect_indices = self.tokenizer.text_to_sequence(aspect)
                aspect_len = np.sum(aspect_indices != 0)
                left_len = min(self.opt.max_seq_len - aspect_len, np.sum(left_indices != 0))
                left_indices = np.concatenate((left_indices[:left_len], np.asarray([0] * (self.opt.max_seq_len - left_len))))
                aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

                idx2graph = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
                dependency_graph = np.pad(idx2graph,
                                          ((0, max(0, self.opt.max_seq_len - idx2graph.shape[0])),
                                           (0, max(0, self.opt.max_seq_len - idx2graph.shape[0]))),
                                          'constant')
                dependency_graph = dependency_graph[:, range(0, self.opt.max_seq_len)]
                dependency_graph = dependency_graph[range(0, self.opt.max_seq_len), :]

                data = {
                    'text_indices': text_indices
                    if 'text_indices' in self.opt.model.inputs else 0,

                    'context_indices': context_indices
                    if 'context_indices' in self.opt.model.inputs else 0,

                    'left_indices': left_indices
                    if 'left_indices' in self.opt.model.inputs else 0,

                    'left_with_aspect_indices': left_with_aspect_indices
                    if 'left_with_aspect_indices' in self.opt.model.inputs else 0,

                    'right_indices': right_indices
                    if 'right_indices' in self.opt.model.inputs else 0,

                    'right_with_aspect_indices': right_with_aspect_indices
                    if 'right_with_aspect_indices' in self.opt.model.inputs else 0,

                    'aspect_indices': aspect_indices
                    if 'aspect_indices' in self.opt.model.inputs else 0,

                    'aspect_boundary': aspect_boundary
                    if 'aspect_boundary' in self.opt.model.inputs else 0,

                    'dependency_graph': dependency_graph
                    if 'dependency_graph' in self.opt.model.inputs else 0,

                    'text_raw': text,
                    'aspect': aspect,
                    'polarity': polarity,
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
