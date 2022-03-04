# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import tqdm
from findfile import find_file
from termcolor import colored
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.core.apc.classic.__bert__.dataset_utils.classic_bert_apc_utils import build_sentiment_window, prepare_input_for_apc
from pyabsa.core.apc.classic.__glove__.dataset_utils.dependency_graph import prepare_dependency_graph, configure_spacy_model
from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, validate_example


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

    def __init__(self, dataset_list, tokenizer, opt):
        configure_spacy_model(opt)

        lines = load_apc_datasets(dataset_list)

        all_data = []
        label_set = set()

        dep_cache_path = os.path.join(os.getcwd(), 'run/{}/dependency_cache/'.format(opt.dataset_name))
        if not os.path.exists(dep_cache_path):
            os.makedirs(dep_cache_path)
        graph_path = prepare_dependency_graph(dataset_list, dep_cache_path, opt.max_seq_len)
        fin = open(graph_path, 'rb')
        idx2graph = pickle.load(fin)

        ex_id = 0

        if len(lines) % 3 != 0:
            print(colored('ERROR: one or more datasets are corrupted, make sure the number of lines in a dataset should be multiples of 3.', 'red'))

        for i in tqdm.tqdm(range(0, len(lines), 3), postfix='building word indices...'):
            if lines[i].count("$T$") > 1:
                continue
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text_raw = text_left + ' ' + aspect + ' ' + text_right
            polarity = lines[i + 2].strip()
            # polarity = int(polarity)
            prepared_inputs = prepare_input_for_apc(opt, tokenizer.tokenizer, text_left, text_right, aspect)

            aspect_position = prepared_inputs['aspect_position']

            text_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + ' ' + aspect + ' ' + text_right + " [SEP]")
            context_indices = tokenizer.text_to_sequence(text_left + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " [SEP]")
            right_indices = tokenizer.text_to_sequence(text_right, reverse=False)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=False)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            left_len = min(opt.max_seq_len - aspect_len, np.sum(left_indices != 0))
            left_indices = np.concatenate((left_indices[:left_len], np.asarray([0] * (opt.max_seq_len - left_len))))
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

            dependency_graph = np.pad(idx2graph[text_raw],
                                      ((0, max(0, opt.max_seq_len - idx2graph[text_raw].shape[0])),
                                       (0, max(0, opt.max_seq_len - idx2graph[text_raw].shape[0]))),
                                      'constant')

            dependency_graph = dependency_graph[:, range(0, opt.max_seq_len)]
            dependency_graph = dependency_graph[range(0, opt.max_seq_len), :]

            validate_example(text_raw, aspect, polarity)

            data = {
                'ex_id': ex_id,

                'text_bert_indices': text_indices
                if 'text_bert_indices' in opt.inputs_cols else 0,

                'context_indices': context_indices
                if 'context_indices' in opt.inputs_cols else 0,

                'left_indices': left_indices
                if 'left_indices' in opt.inputs_cols else 0,

                'left_with_aspect_indices': left_with_aspect_indices
                if 'left_with_aspect_indices' in opt.inputs_cols else 0,

                'right_indices': right_indices
                if 'right_indices' in opt.inputs_cols else 0,

                'right_with_aspect_indices': right_with_aspect_indices
                if 'right_with_aspect_indices' in opt.inputs_cols else 0,

                'aspect_indices': aspect_indices
                if 'aspect_indices' in opt.inputs_cols else 0,

                'aspect_boundary': aspect_boundary
                if 'aspect_boundary' in opt.inputs_cols else 0,

                'aspect_position': aspect_position,

                'dependency_graph': dependency_graph
                if 'dependency_graph' in opt.inputs_cols else 0,

                'polarity': polarity,
            }
            ex_id += 1

            label_set.add(polarity)

            all_data.append(data)

        check_and_fix_labels(label_set, 'polarity', all_data, opt)
        opt.polarities_dim = len(label_set)

        all_data = build_sentiment_window(all_data, tokenizer, opt.similarity_threshold, input_demands=opt.inputs_cols)
        for data in all_data:

            cluster_ids = []
            for pad_idx in range(opt.max_seq_len):
                if pad_idx in data['cluster_ids']:
                    cluster_ids.append(data['polarity'])
                else:
                    cluster_ids.append(-100)
                    # cluster_ids.append(3)

            data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
            data['side_ex_ids'] = np.array(0)
            data['aspect_position'] = np.array(0)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
