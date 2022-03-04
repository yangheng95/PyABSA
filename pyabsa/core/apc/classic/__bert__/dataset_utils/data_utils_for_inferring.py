# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import tqdm
from pyabsa.core.apc.classic.__bert__.dataset_utils.classic_bert_apc_utils import build_sentiment_window

from pyabsa.core.apc.classic.__bert__.dataset_utils.dependency_graph import configure_spacy_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.utils.pyabsa_utils import validate_example
from .dependency_graph import dependency_adj_matrix
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
        configure_spacy_model(opt)

        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        _text = text
        samples = []

        if '!sent!' not in text:
            text += '!sent!'
        text, _, ref_sent = text.partition('!sent!')
        ref_sent = ref_sent.split(',') if ref_sent else None
        text = '[PADDING] ' + text + ' [PADDING]'
        splits = text.split('[ASP]')

        if ref_sent and int((len(splits) - 1) / 2) == len(ref_sent):
            for i in range(0, len(splits) - 1, 2):
                sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                      '[TEMP]' + splits[i + 1] + '[TEMP]', 1).replace('[ASP]', '')
                sample += ' !sent! ' + str(ref_sent[int(i / 2)])
                samples.append(sample.replace('[TEMP]', '[ASP]'))
        elif not ref_sent or int((len(splits) - 1) / 2) != len(ref_sent):
            if not ref_sent:
                print(_text, ' -> No the reference sentiment found')
            else:
                print(_text, ' -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.')

            for i in range(0, len(splits) - 1, 2):
                sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                      '[TEMP]' + splits[i + 1] + '[TEMP]', 1).replace('[ASP]', '')
                samples.append(sample.replace('[TEMP]', '[ASP]'))
        else:
            raise ValueError('Invalid Input:{}'.format(text))

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

        ex_id = 0
        for text in tqdm.tqdm(samples, postfix='building word indices...'):
            try:
                # handle for empty lines in inferring_tutorials dataset_utils
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '!sent!' in text:
                    text, polarity = text.split('!sent!')[0].strip(), text.split('!sent!')[1].strip()
                    polarity = polarity if polarity else LABEL_PADDING
                    text = text.replace('[PADDING]', '')

                else:
                    polarity = str(LABEL_PADDING)

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split('[ASP]')
                text_left = text_left.replace('[PADDING] ', '')
                text_right = text_right.replace(' [PADDING]', '')
                text = text_left + ' ' + aspect + ' ' + text_right
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

                validate_example(text, aspect, polarity)

                data = {
                    'text_indices': text_indices
                    if 'text_indices' in self.opt.inputs_cols else 0,

                    'context_indices': context_indices
                    if 'context_indices' in self.opt.inputs_cols else 0,

                    'left_indices': left_indices
                    if 'left_indices' in self.opt.inputs_cols else 0,

                    'left_with_aspect_indices': left_with_aspect_indices
                    if 'left_with_aspect_indices' in self.opt.inputs_cols else 0,

                    'right_indices': right_indices
                    if 'right_indices' in self.opt.inputs_cols else 0,

                    'right_with_aspect_indices': right_with_aspect_indices
                    if 'right_with_aspect_indices' in self.opt.inputs_cols else 0,

                    'aspect_indices': aspect_indices
                    if 'aspect_indices' in self.opt.inputs_cols else 0,

                    'aspect_boundary': aspect_boundary
                    if 'aspect_boundary' in self.opt.inputs_cols else 0,

                    'dependency_graph': dependency_graph
                    if 'dependency_graph' in self.opt.inputs_cols else 0,

                    'text_raw': text,
                    'aspect': aspect,
                    'polarity': polarity,
                }

                all_data.append(data)
                ex_id += 1

                all_data = build_sentiment_window(all_data, self.tokenizer, self.opt.similarity_threshold, input_demands=self.opt.inputs_cols)
                for data in all_data:

                    cluster_ids = []
                    for pad_idx in range(self.opt.max_seq_len):
                        if pad_idx in data['cluster_ids']:
                            cluster_ids.append(data['polarity'])
                        else:
                            cluster_ids.append(-100)
                            # cluster_ids.append(3)

                    data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
                    data['side_ex_ids'] = np.array(0)
                    data['aspect_position'] = np.array(0)
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
