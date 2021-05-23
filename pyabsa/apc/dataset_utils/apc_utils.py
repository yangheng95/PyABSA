# -*- coding: utf-8 -*-
# file: apc_utils.py
# time: 2021/5/23 0023
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import argparse
import json
import networkx as nx
import numpy as np
import spacy


def parse_experiments(path):
    configs = []

    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for config_id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default=config['model_name'], type=str)
        parser.add_argument('--optimizer', default=config['optimizer'], type=str)
        parser.add_argument('--initializer', default='xavier_uniform_', type=str)
        parser.add_argument('--learning_rate', default=config['learning_rate'], type=float)
        parser.add_argument('--dropout', default=config['dropout'], type=float)
        parser.add_argument('--l2reg', default=config['l2reg'], type=float)
        parser.add_argument('--num_epoch', default=config['num_epoch'], type=int)
        parser.add_argument('--batch_size', default=config['batch_size'], type=int)
        parser.add_argument('--log_step', default=3, type=int)
        parser.add_argument('--logdir', default=config['logdir'], type=str)
        parser.add_argument('--embed_dim', default=768 if 'bert' in config['model_name'] else 300, type=int)
        parser.add_argument('--hidden_dim', default=768 if 'bert' in config['model_name'] else 300, type=int)
        parser.add_argument('--pretrained_bert_name', default='bert-base-uncased' \
            if 'pretrained_bert_name' not in config else config['pretrained_bert_name'], type=str)
        parser.add_argument('--use_bert_spc', default=True \
            if 'use_bert_spc' not in config else config['use_bert_spc'], type=bool)
        parser.add_argument('--use_dual_bert', default=False \
            if 'use_dual_bert' not in config else config['use_dual_bert'], type=bool)
        parser.add_argument('--max_seq_len', default=config['max_seq_len'], type=int)
        parser.add_argument('--polarities_dim', default=3, type=int)
        parser.add_argument('--hops', default=3, type=int)
        parser.add_argument('--SRD', default=config['SRD'], type=int)
        parser.add_argument('--eta', default=config['eta'] if 'eta' in config else -1, type=int)
        parser.add_argument('--lcf', default=config['lcf'], type=str)
        parser.add_argument('--window', default=config['window'], type=str)
        parser.add_argument('--sigma', default=1 if 'sigma' not in config else config['sigma'], type=float)

        configs.append(parser.parse_args())
    return configs


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


class Tokenizer4Bert:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.cls_token
        self.sep_token = tokenizer.sep_token
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    # Group distance to aspect of an original word to its corresponding subword token
    def tokenize(self, text, dep_dist, reverse=False, padding='post', truncating='post'):
        sequence, distances = [], []
        for word, dist in zip(text, dep_dist):
            tokens = self.tokenizer.tokenize(word)
            for jx, token in enumerate(tokens):
                sequence.append(token)
                distances.append(dist)
        sequence = self.tokenizer.convert_tokens_to_ids(sequence)

        if len(sequence) == 0:
            sequence = [0]
            dep_dist = [0]
        if reverse:
            sequence = sequence[::-1]
            dep_dist = dep_dist[::-1]
        sequence = pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        dep_dist = pad_and_truncate(dep_dist, self.max_seq_len, padding=padding, truncating=truncating,
                                    value=self.max_seq_len)

        return sequence, dep_dist

    def get_bert_tokens(self, text):
        return self.tokenizer.tokenize(text)


def copy_side_aspect(direct='left', target=None, source=None):
    # for data_item in ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec']:
    for data_item in ['lcf_vec']:
        target[direct + '_' + data_item] = source[data_item]


def is_similar(s1, s2):
    # some reviews in the datasets are broken so the similarity check is used
    count = 0.
    s1 = list(s1)
    s2 = list(s2)
    s1 = s1[:s1.index(102) if 102 in s1 else len(s1)]
    s2 = s2[:s2.index(102) if 102 in s2 else len(s2)]
    for ids in s1:
        if ids in s2:
            count += 1
    if count / len(s1) >= 0.9 and count / len(s2) >= 0.9:
        return True
    else:
        return False


try:
    # Note that this function is not available for Chinese currently.
    nlp = spacy.load("en_core_web_sm")
except:
    raise RuntimeError('Can not load en_core_web_sm from spacy, maybe you need to download it using:'
                       '\n python -m spacy download en_core_web_sm')


def spacy_tokenize(text):
    doc = nlp(text.strip())
    text_ = []
    for token in doc:
        text_.append(token.lower_)
    return ' '.join(text_)


def calculate_dep_dist(sentence, aspect):
    terms = [a.lower() for a in aspect.split()]
    doc = nlp(sentence)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]:
            term_ids[cnt] = token.i
            cnt += 1

        for child in token.children:
            edges.append(('{}_{}'.format(token.lower_, token.i),
                          '{}_{}'.format(child.lower_, child.i)))

    graph = nx.Graph(edges)

    dist = [0.0] * len(doc)
    text = [''] * len(doc)
    for i, word in enumerate(doc):
        source = '{}_{}'.format(word.lower_, word.i)
        sum = 0
        for term_id, term in zip(term_ids, terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph, source=source, target=target)
            except:
                sum += len(doc)  # No connection between source and target
        dist[i] = sum / len(terms)
        text[i] = word.text
    return text, dist
