# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# author: yangheng<yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import argparse
import json
import torch
import networkx as nx
import spacy


def parse_experiments(path):
    configs = []

    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default=config['model_name'], type=str)
        parser.add_argument('--dataset', default=config['dataset'], type=str, help='twitter, restaurant, laptop')
        parser.add_argument('--optimizer', default=config['optimizer'], type=str)
        parser.add_argument('--initializer', default='xavier_uniform_', type=str)
        parser.add_argument('--learning_rate', default=config['learning_rate'], type=float)
        parser.add_argument('--dropout', default=config['dropout'], type=float)
        parser.add_argument('--l2reg', default=config['l2reg'], type=float)
        parser.add_argument('--num_epoch', default=config['num_epoch'], type=int)
        parser.add_argument('--batch_size', default=config['batch_size'], type=int)
        parser.add_argument('--log_step', default=5, type=int)
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
        parser.add_argument('--lcf', default=config['lcf'], type=str)
        # parser.add_argument('--lcfs', default=False if 'lcfs' not in config else config['lcfs'], choices=[True, False],
        #                     type=bool)
        # parser.add_argument('--lca', default=False if 'lca' not in config else config['lca'], choices=[True, False],
        #                     type=bool)
        # parser.add_argument('--lcp', default=False if 'lcp' not in config else config['lcp'], choices=[True, False],
        #                     type=bool)
        parser.add_argument('--sigma', default=1 if 'sigma' not in config else config['sigma'], type=float)
        parser.add_argument('--repeat', default=config['exp_rounds'], type=bool)

        # The following lines are useless, do not care
        parser.add_argument('--config', default=None, type=str)
        configs.append(parser.parse_args())
    return configs

def build_tokenizer(fnames, max_seq_len, dat_fname=None):
    text = ''
    for fname in fnames:
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

    tokenizer = Tokenizer(max_seq_len)
    tokenizer.fit_on_text(text)
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[len(tokens) - embed_dim:len(tokens)], dtype='float32')
    return word_vec


def load_embedding_matrix(dat_fname):

    return pickle.load(open(dat_fname, 'rb'))


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
        sequence, distances = [],[]
        for word,dist in zip(text,dep_dist):
            tokens = self.tokenizer.tokenize(word)
            for jx,token in enumerate(tokens):
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
        dep_dist = pad_and_truncate(dep_dist, self.max_seq_len, padding=padding, truncating=truncating,value=self.max_seq_len)

        return sequence, dep_dist

class ABSADataset(Dataset):
    input_colses = {
        'bert_base': ['text_raw_bert_indices'],
        'bert_spc': ['text_raw_bert_indices', 'bert_segments_ids'],
        'lca_lstm': ['text_bert_indices', 'text_raw_bert_indices', 'lca_ids', 'lcf_vec'],
        'lca_glove': ['text_bert_indices', 'text_raw_bert_indices', 'lca_ids', 'lcf_vec'],
        'lca_bert': ['text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'lca_ids', 'lcf_vec'],
        'lcf_glove': ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec', ],
        'lcf_bert': ['text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'lcf_vec'],
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tc_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
    }

    def __init__(self, fname, tokenizer, opt):

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        print('buliding word indices...')
        all_data = []

        def get_lca_ids_and_cdm_vec(text_ids, aspect_indices, syntactical_dist=None):
            lca_ids = np.ones((opt.max_seq_len), dtype=np.float32)
            cdm_vec = np.ones((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            aspect_len = np.count_nonzero(aspect_indices) - 2
            aspect_begin = np.argwhere(text_ids == aspect_indices[1])[0]
            mask_begin = aspect_begin - opt.SRD if aspect_begin >= opt.SRD else 0
            # mask_begin = aspect_begin
            for i in range(opt.max_seq_len):
                # if i < mask_begin or i > aspect_begin + aspect_len - 1:
                if i < mask_begin or i > aspect_begin + aspect_len + opt.SRD - 1:
                    lca_ids[i] = 0
                    cdm_vec[i] = np.zeros((opt.embed_dim), dtype=np.float32)
            if 'lcfs' in opt.model_name:
                # Find distance in dependency parsing tree
                raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
                raw_tokens.insert(0, tokenizer.cls_token)
                dist.insert(0, 0)
                raw_tokens.append(tokenizer.sep_token)
                dist.append(0)
                _, distance_to_aspect = tokenizer.tokenize(raw_tokens, dist)
                for i in range(opt.max_seq_len):
                    if syntactical_dist[i] < opt.SRD:
                        lca_ids[i] = 1
                        cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return lca_ids, cdm_vec

        def get_cdw_vec(text_ids, aspect_indices, syntactical_dist=None):
            cdw_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            aspect_len = np.count_nonzero(aspect_indices) - 2
            aspect_begin = np.argwhere(text_ids == aspect_indices[1])[0]
            asp_avg_index = (aspect_begin * 2 + aspect_len) / 2
            text_len = np.flatnonzero(text_ids)[-1] + 1
            if 'lcfs' in opt.model_name:
                # Find distance in dependency parsing tree
                raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
                raw_tokens.insert(0, tokenizer.cls_token)
                dist.insert(0, 0)
                raw_tokens.append(tokenizer.sep_token)
                dist.append(0)
                _, distance_to_aspect = tokenizer.tokenize(raw_tokens, dist)
                for i in range(text_len):
                    if syntactical_dist[i] > opt.SRD:
                        w = 1 - syntactical_dist[i] / text_len
                        cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
                    else:
                        cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            else:
                for i in range(text_len):
                    if abs(i - asp_avg_index) + aspect_len / 2 > opt.SRD:
                        w = 1 - (abs(i - asp_avg_index) + aspect_len / 2 - opt.SRD) / text_len
                        cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
                    else:
                        cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return cdw_vec

        for i in range(0, len(lines)):

            # handle for empty lines in inferring dataset
            if lines[i] is None or '' == lines[i]:
                continue

            # check for given polarity
            if '!sent!' in lines[i]:
                lines[i], polarity = lines[i].split('!sent!')[0].strip(), lines[i].split('!sent!')[1].strip()
                polarity = int(polarity) + 1 if polarity else -999
            else:
                polarity = -999

            text_left = lines[i].replace('$', '').strip()
            text_right = ''

            aspect = lines[i][lines[i].find('$')+1:]
            aspect = aspect[:aspect.find('$')].strip()

            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)

            # Trick: dynamic truncation on input text
            text_left = ' '.join(text_left.split(' ')[int(-(tokenizer.max_seq_len - aspect_len) / 2) - 1:])
            text_right = ' '.join(text_right.split(' ')[:int((tokenizer.max_seq_len - aspect_len) / 2) + 1])
            text_left = ' '.join(text_left.split(' '))
            text_right = ' '.join(text_right.split(' '))
            text_raw = text_left + ' ' + aspect + ' ' + text_right

            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])

            text_raw_indices = tokenizer.text_to_sequence(text_raw)
            text_bert_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            if 'lca' in opt.model_name:
                lca_ids, lcf_vec = get_lca_ids_and_cdm_vec(text_bert_indices, aspect_bert_indices)
                lcf_vec = torch.from_numpy(lcf_vec)
                lca_ids = torch.from_numpy(lca_ids).long()
            elif 'lcf' in opt.model_name:
                if 'cdm' in opt.lcf:
                    _, lcf_vec = get_lca_ids_and_cdm_vec(text_bert_indices, aspect_bert_indices)
                    lcf_vec = torch.from_numpy(lcf_vec)
                elif 'cdw' in opt.lcf:
                    lcf_vec = get_cdw_vec(text_bert_indices, aspect_bert_indices)
                    lcf_vec = torch.from_numpy(lcf_vec)
                elif 'fusion' in opt.lcf:
                    raise NotImplementedError('LCF-Fusion is not recommended due to its low efficiency!')
                else:
                    raise KeyError('Invalid LCF Mode!')

            data = {
                'text_raw': text_raw,
                'aspect': aspect,
                'lca_ids': lca_ids if 'lca_ids' in ABSADataset.input_colses[opt.model_name] else 0,
                'lcf_vec': lcf_vec if 'lcf_vec' in ABSADataset.input_colses[opt.model_name] else 0,
                'text_bert_indices': text_bert_indices if 'text_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'bert_segments_ids': bert_segments_ids if 'bert_segments_ids' in ABSADataset.input_colses[opt.model_name] else 0,

                'aspect_bert_indices': aspect_bert_indices if 'aspect_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'text_raw_indices': text_raw_indices if 'text_raw_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'aspect_indices': aspect_indices if 'aspect_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'text_left_indices': text_left_indices if 'text_left_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'aspect_in_text': aspect_in_text if 'aspect_in_text' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_raw_without_aspect_indices': text_raw_without_aspect_indices
                if 'text_raw_without_aspect_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_left_with_aspect_indices': text_left_with_aspect_indices
                if 'text_left_with_aspect_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_right_indices': text_right_indices
                if 'text_right_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_right_with_aspect_indices': text_right_with_aspect_indices
                if 'text_right_with_aspect_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_raw_bert_indices': text_raw_bert_indices
                if 'text_raw_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'polarity': polarity,
            }
            for _, item in enumerate(data):
                data[item] = torch.tensor(data[item]) if type(item) is not str else data[item]
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Note that this function is not suitable for both Chinese and English currently.
nlp = spacy.load("en_core_web_sm")
def calculate_dep_dist(sentence,aspect):
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
            edges.append(('{}_{}'.format(token.lower_,token.i),
                          '{}_{}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    for i,word in enumerate(doc):
        source = '{}_{}'.format(word.lower_,word.i)
        sum = 0
        for term_id,term in zip(term_ids,terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph,source=source,target=target)
            except:
                sum += len(doc) # No connection between source and target
        dist[i] = sum/len(terms)
        text[i] = word.text
    return text,dist