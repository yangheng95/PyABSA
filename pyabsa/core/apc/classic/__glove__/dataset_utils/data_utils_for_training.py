# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import tqdm
from findfile import find_file
from google_drive_downloader.google_drive_downloader import GoogleDriveDownloader as gdd
from termcolor import colored
from torch.utils.data import Dataset

from .dependency_graph import prepare_dependency_graph
from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, validate_example
from ...__glove__.dataset_utils.dependency_graph import configure_spacy_model


def prepare_glove840_embedding(glove_path):
    glove840_id = '1G-vd6W1oF9ByyJ-pzp9dcqKnr_plh4Em'
    if not os.path.exists(glove_path):
        os.mkdir(glove_path)
    elif os.path.isfile(glove_path):
        return glove_path
    elif os.path.isdir(glove_path):
        embedding_file = None
        dir_path = os.path.dirname(glove_path)
        if find_file(dir_path, 'glove.42B.300d.txt', exclude_key='.zip'):
            embedding_file = find_file(dir_path, 'glove.42B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.840B.300d.txt', exclude_key='.zip'):
            embedding_file = find_file(dir_path, 'glove.840B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip'):
            embedding_file = find_file(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip')

        if embedding_file:
            print('Find potential embedding files: {}'.format(embedding_file))
            return embedding_file
        zip_glove_path = os.path.join(glove_path, 'glove.840B.300d.txt.zip')
        print('No GloVe embedding found at {},'
              ' downloading glove.840B.300d.txt (2GB transfer size / 5.5GB unzip size)...'.format(glove_path))
        gdd.download_file_from_google_drive(file_id=glove840_id,
                                            dest_path=zip_glove_path,
                                            unzip=True,
                                            showsize=True
                                            )
        glove_path = find_file(glove_path, 'txt', exclude_key='.zip')
    return glove_path


def build_tokenizer(dataset_list, max_seq_len, dat_fname, opt):
    if os.path.exists(os.path.join(opt.dataset_name, dat_fname)):
        print('Loading tokenizer on {}'.format(os.path.join(opt.dataset_name, dat_fname)))
        tokenizer = pickle.load(open(os.path.join(opt.dataset_name, dat_fname), 'rb'))
    else:
        text = ''
        for dataset_type in dataset_list:
            for file in dataset_list[dataset_type]:
                fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                lines = fin.readlines()
                fin.close()
                for i in range(0, len(lines), 3):
                    text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                    aspect = lines[i + 1].lower().strip()
                    text_raw = text_left + " " + aspect + " " + text_right
                    text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(os.path.join(opt.dataset_name, dat_fname), 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in tqdm.tqdm(fin.readlines(), postfix='Loading embedding file...'):
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, opt):
    if os.path.exists(os.path.join(opt.dataset_name, dat_fname)):
        print('Loading cached embedding_matrix for {}'.format(os.path.join(opt.dataset_name, dat_fname)))
        embedding_matrix = pickle.load(open(os.path.join(opt.dataset_name, dat_fname), 'rb'))
    else:
        print('Extracting embedding_matrix for {}'.format(dat_fname))
        glove_path = prepare_glove840_embedding(opt.dataset_name)
        opt.glove = glove_path
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(glove_path, word2idx=word2idx, embed_dim=embed_dim)

        for word, i in tqdm.tqdm(word2idx.items(), postfix='Building embedding_matrix {}'.format(dat_fname)):
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(os.path.join(opt.dataset_name, dat_fname), 'wb'))
    return embedding_matrix


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


class GloVeABSADataset(Dataset):

    def __init__(self, dataset_list, tokenizer, opt):
        configure_spacy_model(opt)
        lines = load_apc_datasets(dataset_list)
        all_data = []
        label_set = set()

        dep_cache_path = os.path.join(os.getcwd(), '{}_dependency_cache'.format(opt.dataset_name))
        if not os.path.exists(dep_cache_path):
            os.mkdir(dep_cache_path)
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

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            left_indices = tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
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

                'text_indices': text_indices
                if 'text_indices' in opt.inputs_cols else 0,

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

                'dependency_graph': dependency_graph
                if 'dependency_graph' in opt.inputs_cols else 0,

                'polarity': polarity,
            }
            ex_id += 1

            label_set.add(polarity)

            all_data.append(data)

        check_and_fix_labels(label_set, 'polarity', all_data, opt)
        opt.polarities_dim = len(label_set)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
