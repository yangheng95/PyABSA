# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import zipfile

import gdown
import numpy as np
import tqdm
from findfile import find_file, find_cwd_file, find_files
from torch.utils.data import Dataset

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels

LABEL_PADDING = -999


def prepare_glove840_embedding(glove_path):
    glove840_id = '1G-vd6W1oF9ByyJ-pzp9dcqKnr_plh4Em'
    if os.path.exists(glove_path) and os.path.isfile(glove_path):
        return glove_path
    else:
        embedding_files = []
        dir_path = os.getenv('$HOME') if os.getenv('$HOME') else os.getcwd()

        if find_file(dir_path, 'glove.42B.300d.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.42B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.840B.300d.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.840B.300d.txt', exclude_key='.zip')
        elif find_file(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip'):
            embedding_files += find_files(dir_path, 'glove.twitter.27B.txt', exclude_key='.zip')

        if embedding_files:
            print('Find embedding file: {}, use the first: {}'.format(embedding_files, embedding_files[0]))
            return embedding_files[0]

        else:
            zip_glove_path = os.path.join(os.path.dirname(glove_path), 'glove.840B.300d.zip')
            print('No GloVe embedding found at {},'
                  ' downloading glove.840B.300d.txt (2GB will be downloaded / 5.5GB after unzip)...'.format(glove_path))
            gdown.download(id=glove840_id, output=zip_glove_path)

        if find_cwd_file('glove.840B.300d.zip'):
            with zipfile.ZipFile(find_cwd_file('glove.840B.300d.zip'), 'r') as z:
                z.extractall()
            print('Zip file extraction Done.')

        return prepare_glove840_embedding(glove_path)


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


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in tqdm.tqdm(fin, postfix='Loading embedding file...'):
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname, opt):
    if not os.path.exists('run'):
        os.makedirs('run')
    embed_matrix_path = 'run/{}'.format(os.path.join(opt.dataset_name, dat_fname))
    if os.path.exists(embed_matrix_path):
        print('Loading cached embedding_matrix for {}'.format(embed_matrix_path))
        embedding_matrix = pickle.load(open(embed_matrix_path, 'rb'))
    else:
        glove_path = prepare_glove840_embedding(embed_matrix_path)
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(glove_path, word2idx=word2idx, embed_dim=embed_dim)

        for word, i in tqdm.tqdm(word2idx.items(), postfix='Building embedding_matrix {}'.format(dat_fname)):
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embed_matrix_path, 'wb'))
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


class GloVeClassificationDataset(Dataset):
    glove_input_colses = {
        'lstm': ['text_indices']
    }

    def __init__(self, dataset_list, tokenizer, opt):
        lines = load_apc_datasets(dataset_list)

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='building word indices...'):
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
