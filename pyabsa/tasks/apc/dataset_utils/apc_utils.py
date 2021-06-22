# -*- coding: utf-8 -*-
# file: apc_utils.py
# time: 2021/5/23 0023
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import warnings
import spacy
import termcolor
import networkx as nx
import numpy as np
import os

SENTIMENT_PADDING = -999


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
        self.bos_token = tokenizer.bos_token if tokenizer.bos_token else '[CLS]'
        self.eos_token = tokenizer.eos_token if tokenizer.eos_token else '[SEP]'
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else 101
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 102
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def get_bert_tokens(self, text):
        return self.tokenizer.tokenize(text)


def syntax_distance_alignment(tokens, dist, max_seq_len, tokenizer):
    text = tokens[:]
    dep_dist = dist[:]
    bert_tokens = tokenizer.tokenize(' '.join(text))
    _bert_tokens = bert_tokens[:]
    align_dist = []
    if bert_tokens != text:
        while text or bert_tokens:
            if text[0] == ' ' or text[0] == '\xa0':  # bad case handle
                text = text[1:]
                dep_dist = dep_dist[1:]
            elif text[0] == bert_tokens[0]:
                text = text[1:]
                bert_tokens = bert_tokens[1:]
                align_dist.append(dep_dist[0])
                dep_dist = dep_dist[1:]
            elif len(text[0]) < len(bert_tokens[0]):
                tmp_str = text[0]
                while len(tmp_str) < len(bert_tokens[0]):
                    text = text[1:]
                    tmp_str += text[0]
                    dep_dist = dep_dist[1:]
                align_dist.append(dep_dist[0])
                dep_dist = dep_dist[1:]
                text = text[1:]
                bert_tokens = bert_tokens[1:]
            elif len(text[0]) > len(bert_tokens[0]):
                tmp_tokens = tokenizer.tokenize(text[0])
                for jx, tmp_token in enumerate(tmp_tokens):
                    align_dist.append(dep_dist[0])

                text = text[1:]
                dep_dist = dep_dist[1:]
                bert_tokens = bert_tokens[len(tmp_tokens):]
            else:
                text = text[1:]
                bert_tokens = bert_tokens[1:]
                align_dist.append(dep_dist[0])
                dep_dist = dep_dist[1:]

    else:
        align_dist = dep_dist

    align_dist = pad_and_truncate(align_dist, max_seq_len, value=max_seq_len)
    return align_dist


# Group distance to aspect of an original word to its corresponding subword token
def pad_syntax_based_srd(text, dep_dist, tokenizer, opt):
    sequence, distances = [], []
    for word, dist in zip(text, dep_dist):
        tokens = tokenizer.tokenize(word)
        for jx, token in enumerate(tokens):
            sequence.append(token)
            distances.append(dist)
    sequence = tokenizer.convert_tokens_to_ids(sequence)
    sequence = pad_and_truncate(sequence, opt.max_seq_len)
    dep_dist = pad_and_truncate(dep_dist, opt.max_seq_len, value=opt.max_seq_len)

    return sequence, dep_dist


def load_datasets(fname):
    lines = []
    if isinstance(fname, str):
        fname = [fname]

    for f in fname:
        print('loading: {}'.format(f))
        fin = open(f, 'r', encoding='utf-8')
        lines.extend(fin.readlines())
        fin.close()
    return lines


def prepare_input_for_apc(opt, tokenizer, text_left, text_right, aspect):
    if hasattr(opt, 'dynamic_truncate') and opt.dynamic_truncate:
        _max_seq_len = opt.max_seq_len - len(aspect.split())
        text_left = text_left.split(' ')
        text_right = text_right.split(' ')
        if _max_seq_len < (len(text_left) + len(text_right)):
            cut_len = len(text_left) + len(text_right) - _max_seq_len
            if len(text_left) > len(text_right):
                text_left = text_left[cut_len:]
            else:
                text_right = text_right[:len(text_right) - cut_len]
        text_left = ' '.join(text_left)
        text_right = ' '.join(text_right)
    bos_token = tokenizer.tokenizer.bos_token if tokenizer.tokenizer.bos_token else '[CLS]'
    eos_token = tokenizer.tokenizer.eos_token if tokenizer.tokenizer.eos_token else '[SEP]'

    text_raw = text_left + ' ' + aspect + ' ' + text_right
    text_spc = bos_token + ' ' + text_raw + ' ' + eos_token + ' ' + aspect + ' ' + eos_token
    text_bert_indices = tokenizer.text_to_sequence(text_spc)
    text_raw_bert_indices = tokenizer.text_to_sequence(bos_token + ' ' + text_raw + ' ' + eos_token)
    aspect_bert_indices = tokenizer.text_to_sequence(aspect)

    aspect_begin = len(tokenizer.tokenizer.tokenize(bos_token + ' ' + text_left))
    if 'lcfs' in opt.model_name or opt.use_syntax_based_SRD:
        syntactical_dist = get_syntax_distance(text_raw, aspect, tokenizer.tokenizer, opt)
    else:
        syntactical_dist = None

    lca_ids, lcf_cdm_vec = get_lca_ids_and_cdm_vec(opt, text_bert_indices, aspect_bert_indices,
                                                   aspect_begin, syntactical_dist)

    lcf_cdw_vec = get_cdw_vec(opt, text_bert_indices, aspect_bert_indices,
                              aspect_begin, syntactical_dist)

    inputs = {
        'text_raw': text_raw,
        'text_spc': text_spc,
        'aspect': aspect,
        'text_bert_indices': text_bert_indices,
        'text_raw_bert_indices': text_raw_bert_indices,
        'aspect_bert_indices': aspect_bert_indices,
        'lca_ids': lca_ids,
        'lcf_cdm_vec': lcf_cdm_vec,
        'lcf_cdw_vec': lcf_cdw_vec,
    }

    return inputs


def get_syntax_distance(text_raw, aspect, tokenizer, opt):
    # Find distance in dependency parsing tree
    if isinstance(text_raw, list):
        text_raw = ' '.join(text_raw)

    if isinstance(aspect, list):
        aspect = ' '.join(aspect)
    try:
        raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
    except Exception as e:
        print(e)
        raise RuntimeError('Are you using syntax-based SRD on a dataset containing Chinese text?')
    raw_tokens.insert(0, tokenizer.bos_token)
    dist.insert(0, max(dist))
    raw_tokens.append(tokenizer.eos_token)
    dist.append(max(dist))
    # the following two functions are both designed to calculate syntax-based distances
    if opt.srd_alignment:
        syntactical_dist = syntax_distance_alignment(raw_tokens, dist, opt.max_seq_len, tokenizer)
    else:
        syntactical_dist = pad_syntax_based_srd(raw_tokens, dist, tokenizer, opt)[1]
    return syntactical_dist


def get_lca_ids_and_cdm_vec(opt, bert_spc_indices, aspect_indices, aspect_begin, syntactical_dist=None):
    SRD = opt.SRD
    lca_ids = np.zeros((opt.max_seq_len), dtype=np.int64)
    cdm_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
    aspect_len = np.count_nonzero(aspect_indices)
    text_len = np.count_nonzero(bert_spc_indices) - np.count_nonzero(aspect_indices) - 1
    if syntactical_dist is not None:
        for i in range(min(text_len, opt.max_seq_len)):
            if syntactical_dist[i] <= SRD:
                lca_ids[i] = 1
                cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
    else:
        local_context_begin = max(0, aspect_begin - SRD)
        local_context_end = min(aspect_begin + aspect_len + SRD - 1, opt.max_seq_len)
        for i in range(min(text_len, opt.max_seq_len)):
            if local_context_begin <= i <= local_context_end:
                lca_ids[i] = 1
                cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
    return lca_ids, cdm_vec


def get_cdw_vec(opt, bert_spc_indices, aspect_indices, aspect_begin, syntactical_dist=None):
    SRD = opt.SRD
    cdw_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
    aspect_len = np.count_nonzero(aspect_indices)
    text_len = np.count_nonzero(bert_spc_indices) - np.count_nonzero(aspect_indices) - 1
    if syntactical_dist is not None:
        for i in range(min(text_len, opt.max_seq_len)):
            if syntactical_dist[i] > SRD:
                w = 1 - syntactical_dist[i] / text_len
                cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
            else:
                cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
    else:
        local_context_begin = max(0, aspect_begin - SRD)
        local_context_end = min(aspect_begin + aspect_len + SRD - 1, opt.max_seq_len)
        for i in range(min(text_len, opt.max_seq_len)):
            if i < local_context_begin:
                w = 1 - (local_context_begin - i) / text_len
                cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
            elif local_context_begin <= i <= local_context_end:
                cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            elif i > local_context_end:
                w = 1 - (i - local_context_end) / text_len
                cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
    return cdw_vec


def get_asp_index(text_ids, aspect_indices):
    try:
        aspect_len = np.count_nonzero(aspect_indices)
        aspect_indices = aspect_indices[0: aspect_len]
        for i in range(len(text_ids)):
            for j in range(len(aspect_indices)):
                if text_ids[i + j] == aspect_indices[j] and j == len(aspect_indices) - 1:
                    return i
                elif text_ids[i + j] != aspect_indices[j]:
                    break
    except:
        return -1
    return -1


def build_spc_mask_vec(opt, text_ids):
    spc_mask_vec = (text_ids > 0)
    spc_mask_vec = np.array(
        [np.ones((opt.embed_dim))
         if vec_i else np.zeros((opt.embed_dim))
         for vec_i in spc_mask_vec]
    ).astype(np.float32)
    return spc_mask_vec


def build_sentiment_window(examples, tokenizer, similarity_threshold=0.95):
    copy_side_aspect('left', examples[0], examples[0])
    for idx in range(1, len(examples)):
        if is_similar(examples[idx - 1]['text_bert_indices'],
                      examples[idx]['text_bert_indices'],
                      tokenizer=tokenizer,
                      similarity_threshold=similarity_threshold):
            copy_side_aspect('right', examples[idx - 1], examples[idx])
            copy_side_aspect('left', examples[idx], examples[idx - 1])
        else:
            copy_side_aspect('right', examples[idx - 1], examples[idx - 1])
            copy_side_aspect('left', examples[idx], examples[idx])
    copy_side_aspect('right', examples[-1], examples[-1])
    return examples


def copy_side_aspect(direct='left', target=None, source=None):
    for data_item in ['lcf_vec']:
        target[direct + '_' + data_item] = source[data_item]


# buggy code
# def is_similar(s1, s2, tokenizer, similarity_threshold):
#     # some reviews in the datasets are broken and can not use s1 == s2 to distinguish
#     # the same text which contains multiple aspects, so the similarity check is used
#     # similarity check is based on the observation and analysis of datasets
#     count = 0.
#     s1 = list(s1)
#     s2 = list(s2)
#     s1 = s1[:s1.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s1 else len(s1)]
#     s2 = s2[:s2.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s2 else len(s2)]
#     len1 = len(s1)
#     len2 = len(s2)
#     for i, ids in enumerate(s1):
#         if ids in s2:
#             count += 1
#     if count / len1 >= similarity_threshold and count / len2 >= similarity_threshold:
#         return True
#     else:
#         return False


def is_similar(s1, s2, tokenizer, similarity_threshold):
    # some reviews in the datasets are broken and can not use s1 == s2 to distinguish
    # the same text which contains multiple aspects, so the similarity check is used
    # similarity check is based on the observation and analysis of datasets
    if abs(np.count_nonzero(s1) - np.count_nonzero(s2)) > 5:
        return False
    count = 0.
    s1 = list(s1)
    s2 = list(s2)
    s1 = s1[:s1.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s1 else len(s1)]
    s2 = s2[:s2.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in s2 else len(s2)]
    len1 = len(s1)
    len2 = len(s2)
    while s1 and s2:
        if s1[-1] in s2:
            count += 1
            s2.remove(s1[-1])
        s1.remove(s1[-1])

    if count / len1 >= similarity_threshold and count / len2 >= similarity_threshold:
        return True
    else:
        return False


def is_same(s1, s2, tokenizer):
    eos_idx1 = np.min(np.where(s1 == tokenizer.eos_token_id))
    eos_idx2 = np.min(np.where(s2 == tokenizer.eos_token_id))
    s1 = s1[:eos_idx1] if np.count_nonzero(s1) < tokenizer.max_seq_len else s1
    s2 = s2[:eos_idx2] if np.count_nonzero(s2) < tokenizer.max_seq_len else s2
    s1 = list(s1)
    s2 = list(s2)
    return s1 == s2


try:
    nlp = spacy.load("en_core_web_sm")
except:
    print('Can not load en_core_web_sm from spacy, try to download it in order to parse syntax tree:',
          termcolor.colored('\npython -m spacy download en_core_web_sm', 'green'))
    try:
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load("en_core_web_sm")
    except:
        raise RuntimeError('Download failed, you can download en_core_web_sm manually.')

def calculate_dep_dist(sentence, aspect):
    terms = [a.lower() for a in aspect.split()]
    try:
        doc = nlp(sentence)
    except NameError as e:
        raise RuntimeError('Fail to load nlp model, maybe you forget to download en_core_web_sm')
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
