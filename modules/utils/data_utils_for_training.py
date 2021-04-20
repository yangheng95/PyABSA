# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# modified: yangheng<yangheng@m.scnu.edu.cn>

import argparse
import json
import os
import pickle

import networkx as nx
import numpy as np
import spacy
import torch
from torch.utils.data import Dataset


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
        parser.add_argument('--lcf', default=config['lcf'], type=str)
        parser.add_argument('--window', default=config['window'], type=str)
        parser.add_argument('--distance_aware_window', default=config['distance_aware_window'], type=str)
        parser.add_argument('--sigma', default=1 if 'sigma' not in config else config['sigma'], type=float)
        parser.add_argument('--repeat', default=config['exp_rounds'], type=bool)

        # The following lines are useless, do not care
        parser.add_argument('--config', default=None, type=str)
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


class ABSADataset(Dataset):
    input_colses = {
        'bert_base': ['text_raw_bert_indices'],
        'bert_spc': ['text_raw_bert_indices', 'bert_segments_ids'],
        'lca_bert': ['text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'lca_ids', 'lcf_vec'],
        'lcf_bert': ['text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'lcf_vec'],
        'slide_lcf_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
        'slide_lcfs_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
        'lcfs_bert': ['text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'lcf_vec'],
    }

    def __init__(self, fname, tokenizer, opt):
        ABSADataset.opt = opt
        def get_lca_ids_and_cdm_vec(text_ids, aspect_indices):
            lca_ids = np.ones((opt.max_seq_len), dtype=np.float32)
            cdm_vec = np.ones((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            aspect_len = np.count_nonzero(aspect_indices) - 2

            if 'lcfs' in opt.model_name:
                # Find distance in dependency parsing tree
                raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
                raw_tokens.insert(0, tokenizer.cls_token)
                dist.insert(0, 0)
                raw_tokens.append(tokenizer.sep_token)
                dist.append(0)
                syntactical_dist = tokenizer.tokenize(raw_tokens, dist)[1]
                for i in range(opt.max_seq_len):
                    if syntactical_dist[i] < opt.SRD:
                        lca_ids[i] = 1
                        cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            else:
                try:
                    aspect_begin = np.argwhere(text_ids == aspect_indices[1])[0]
                except:
                    return np.ones((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
                mask_begin = aspect_begin - opt.SRD if aspect_begin >= opt.SRD else 0
                for i in range(opt.max_seq_len):
                    if i < mask_begin or i > aspect_begin + aspect_len + opt.SRD - 1:
                        lca_ids[i] = 0
                        cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return lca_ids, cdm_vec

        def get_cdw_vec(text_ids, aspect_indices):
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
                syntactical_dist = tokenizer.tokenize(raw_tokens, dist)[1]
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

        def get_asp_index(text_ids, aspect_indices):
            aspect_len = np.count_nonzero(aspect_indices) - 2
            aspect_begin = np.argwhere(text_ids == aspect_indices[1])[0]
            asp_avg_index = (aspect_begin * 2 + aspect_len) / 2

            return asp_avg_index

        def build_spc_mask_vec(text_ids):
            spc_mask_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            for i in range(len(text_ids)):
                if text_ids[i] != 0:
                    spc_mask_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return spc_mask_vec

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        print('buliding word indices...')
        all_data = []

        for i in range(0, len(lines), 3):

            text_left, _, text_right = [s for s in lines[i].partition("$T$")]
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1
            aspect = lines[i + 1].lower().strip()

            # dynamic truncation on input text
            text_left = ' '.join(text_left.split(' ')[int(-(tokenizer.max_seq_len - len(aspect.split())) / 2) - 1:])
            text_right = ' '.join(text_right.split(' ')[:int((tokenizer.max_seq_len - len(aspect.split())) / 2) + 1])
            text_left = ' '.join(text_left.split(' '))
            text_right = ' '.join(text_right.split(' '))
            text_raw = text_left + ' ' + aspect + ' ' + text_right

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_raw + ' [SEP] ' + aspect + " [SEP]")
            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_raw + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            text_raw_indices = tokenizer.text_to_sequence(text_raw)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

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
                'asp_index': get_asp_index(text_bert_indices, aspect_bert_indices),
                'lca_ids': lca_ids if 'lca_ids' in ABSADataset.input_colses[opt.model_name] else 0,
                'lcf_vec': lcf_vec if 'lcf_vec' in ABSADataset.input_colses[opt.model_name] else 0,
                'spc_mask_vec': build_spc_mask_vec(text_raw_bert_indices) if 'spc_mask_vec' in ABSADataset.input_colses[
                    opt.model_name] else 0,
                'text_bert_indices': text_bert_indices if 'text_bert_indices' in ABSADataset.input_colses[
                    opt.model_name] else 0,
                'bert_segments_ids': bert_segments_ids if 'bert_segments_ids' in ABSADataset.input_colses[
                    opt.model_name] else 0,
                'aspect_bert_indices': aspect_bert_indices if 'aspect_bert_indices' in ABSADataset.input_colses[
                    opt.model_name] else 0,

                'text_raw_indices': text_raw_indices if 'text_raw_indices' in ABSADataset.input_colses[
                    opt.model_name] else 0,
                'aspect_indices': aspect_indices if 'aspect_indices' in ABSADataset.input_colses[opt.model_name] else 0,
                'text_left_indices': text_left_indices if 'text_left_indices' in ABSADataset.input_colses[
                    opt.model_name] else 0,
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

        if 'slide' in opt.model_name:
            ABSADataset.copy_side_aspect('left', all_data[0], all_data[0])
            for idx in range(1, len(all_data)):
                if ABSADataset.is_similar(all_data[idx - 1]['text_bert_indices'], all_data[idx]['text_bert_indices']):
                    ABSADataset.copy_side_aspect('right', all_data[idx - 1], all_data[idx])
                    ABSADataset.copy_side_aspect('left', all_data[idx], all_data[idx - 1])
                else:
                    ABSADataset.copy_side_aspect('right', all_data[idx - 1], all_data[idx - 1])
                    ABSADataset.copy_side_aspect('left', all_data[idx], all_data[idx])
            ABSADataset.copy_side_aspect('right', all_data[-1], all_data[-1])

        self.data = all_data

    @staticmethod
    def copy_side_aspect(direct='left', target=None, source=None):
        for data_item in ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec']:
            target[direct + '_' + data_item] = source[data_item]
        a = abs(source['asp_index'] - target['asp_index'])
        b = np.count_nonzero(target['text_bert_indices'])
        c = 1 - a / b
        target[direct + '_asp_dist_w'] = np.ones((target['lcf_vec'].shape[0], target['lcf_vec'].shape[1]),
                                                 dtype=np.float32)
        for i in range(len(target[direct + '_asp_dist_w'])):
            if ABSADataset.opt.distance_aware_window:
                target[direct + '_asp_dist_w'][i] = c * np.ones((target['lcf_vec'].shape[1]), dtype=np.float32)
            else:
                target[direct + '_asp_dist_w'][i] = np.ones((target['lcf_vec'].shape[1]), dtype=np.float32)

    @staticmethod
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

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Note that this function is not available for Chinese currently.
nlp = spacy.load("en_core_web_sm")


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
    text = [0] * len(doc)
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
