# -*- coding: utf-8 -*-
# file: tokenizer_class.py
# time: 03/11/2022 21:44
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os
import pickle
from typing import Union, List

import numpy as np
import tqdm
from termcolor import colored
from transformers import AutoTokenizer

from pyabsa.utils.file_utils.file_utils import prepare_glove840_embedding
from pyabsa.utils.pyabsa_utils import fprint


class Tokenizer(object):
    def __init__(self, config):
        self.config = config
        self.max_seq_len = self.config.max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1
        self.pre_tokenizer = None
        self.pad_token_id = 0
        self.unk_token_id = 0
        self.cls_token_id = 0
        self.sep_token_id = 0
        self.mask_token_id = 0

    @staticmethod
    def build_tokenizer(config, cache_path=None, pre_tokenizer=None, **kwargs):
        Tokenizer.pre_tokenizer = pre_tokenizer
        dataset_name = os.path.basename(config.dataset_name)
        if not os.path.exists('run/{}'.format(dataset_name)):
            os.makedirs('run/{}'.format(dataset_name))
        tokenizer_path = 'run/{}/{}'.format(dataset_name, cache_path)
        if cache_path and os.path.exists(tokenizer_path) and not config.overwrite_cache:
            config.logger.info('Loading tokenizer on {}'.format(tokenizer_path))
            tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        else:
            words = set()
            if hasattr(config, 'dataset_file'):
                config.logger.info('Building tokenizer for {} on {}'.format(config.dataset_file, tokenizer_path))
                for dataset_type in config.dataset_file:
                    for file in config.dataset_file[dataset_type]:
                        fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                        lines = fin.readlines()
                        fin.close()
                        for i in range(0, len(lines)):
                            if pre_tokenizer:
                                words.update(pre_tokenizer.tokenize(lines[i].strip()))
                            else:
                                words.update(lines[i].strip().split())
            elif hasattr(config, 'dataset_dict'):
                config.logger.info('Building tokenizer for {} on {}'.format(config.dataset_name, tokenizer_path))
                for dataset_type in ['train', 'test', 'valid']:
                    for i, data in enumerate(config.dataset_dict[dataset_type]):
                        if pre_tokenizer:
                            words.update(pre_tokenizer.tokenize(data['data']))
                        else:
                            words.update(data['data'].split())
            tokenizer = Tokenizer(config)
            tokenizer.pre_tokenizer = pre_tokenizer
            tokenizer.fit_on_text(list(words))
            if config.cache_dataset:
                pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

        return tokenizer

    def fit_on_text(self, text: Union[str, List[str]], **kwargs):
        if isinstance(text, str):
            if self.pre_tokenizer:
                words = self.pre_tokenizer.tokenize(text)
            else:
                words = text.split()
        else:
            words = text
        for word in words:
            if self.config.do_lower_case:
                word = word.lower()
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text: Union[str, List[str]], padding='max_length', **kwargs):
        if isinstance(text, str):
            if self.config.do_lower_case:
                text = text.lower()
            if self.pre_tokenizer:
                words = self.pre_tokenizer.tokenize(text)
            else:
                words = text.split()
            sequence = [self.word2idx[w] if w in self.word2idx else 0 for w in words]
            if len(sequence) == 0:
                sequence = [0]
            if kwargs.get('reverse', False):
                sequence = sequence[::-1]
            if padding == 'max_length':
                return pad_and_truncate(sequence, self.max_seq_len, self.pad_token_id)
            else:
                return sequence

        elif isinstance(text, list):
            sequences = []
            for t in text:
                sequences.append(self.text_to_sequence(t, **kwargs))
            return sequences
        else:
            raise ValueError('text_to_sequence only support str or list of str')

    def sequence_to_text(self, sequence):
        words = [self.idx2word[idx] if idx in self.idx2word else '<unk>' for idx in sequence]
        return ' '.join(words)


class PretrainedTokenizer:
    def __init__(self, config, **kwargs):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_bert, **kwargs)
        self.max_seq_len = self.config.max_seq_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def text_to_sequence(self, text, padding='max_length', return_tensors=None, **kwargs):
        return self.tokenizer.encode(text,
                                     truncation=True,
                                     padding=padding,
                                     max_length=self.max_seq_len,
                                     return_tensors=return_tensors,
                                     **kwargs)

    def sequence_to_text(self, sequence, **kwargs):
        return self.tokenizer.decode(sequence, **kwargs)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

    def convert_tokens_to_ids(self, return_tensors=None, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(return_tensors, **kwargs)

    def convert_ids_to_tokens(self, ids, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(ids, **kwargs)


def build_embedding_matrix(config, tokenizer, cache_path=None):
    if not os.path.exists('run/{}'.format(config.dataset_name)):
        os.makedirs('run/{}'.format(config.dataset_name))
    embed_matrix_path = 'run/{}'.format(os.path.join(config.dataset_name, cache_path))
    if cache_path and os.path.exists(embed_matrix_path) and not config.overwrite_cache:
        fprint(colored('Loading cached embedding_matrix from {} (Please remove all cached files if there is any problem!)'.format(embed_matrix_path), 'green'))
        embedding_matrix = pickle.load(open(embed_matrix_path, 'rb'))
    else:
        glove_path = prepare_glove840_embedding(embed_matrix_path, config.embed_dim, config=config)
        embedding_matrix = np.zeros((len(tokenizer.word2idx) + 1, config.embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(glove_path, word2idx=tokenizer.word2idx, embed_dim=config.embed_dim)

        for word, i in tqdm.tqdm(tokenizer.word2idx.items(), postfix=colored('Building embedding_matrix {}'.format(cache_path), 'yellow')):
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        if config.cache_dataset:
            pickle.dump(embedding_matrix, open(embed_matrix_path, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, max_seq_len, value, **kwargs):
    if len(sequence) > max_seq_len:
        sequence = sequence[:max_seq_len]
    else:
        sequence = sequence + [value] * (max_seq_len - len(sequence))
    return sequence


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in tqdm.tqdm(fin.readlines(), postfix='Loading embedding file...'):
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec
