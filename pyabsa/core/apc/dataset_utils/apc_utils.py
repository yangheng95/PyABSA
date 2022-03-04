# -*- coding: utf-8 -*-
# file: apc_utils.py
# time: 2021/5/23 0023
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os

import copy
import networkx as nx
import numpy as np
import spacy
import termcolor

LABEL_PADDING = -999


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


def syntax_distance_alignment(tokens, dist, max_seq_len, tokenizer):
    text = tokens[:]
    dep_dist = dist[:]
    bert_tokens = tokenizer.tokenize(' '.join(text))
    _bert_tokens = bert_tokens[:]
    align_dist = []
    if bert_tokens != text:
        while text or bert_tokens:
            try:
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
            except:
                align_dist = pad_and_truncate(align_dist, max_seq_len, value=max_seq_len)
                return align_dist
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


def load_apc_datasets(fname):
    lines = []
    if isinstance(fname, str):
        fname = [fname]

    for f in fname:
        print('loading: {}'.format(f))
        fin = open(f, 'r', encoding='utf-8')
        _lines_ = fin.readlines()
        lines.extend(_lines_)
        fin.close()
    return lines


def prepare_input_for_apc(opt, tokenizer, text_left, text_right, aspect, input_demands):
    if hasattr(opt, 'dynamic_truncate') and opt.dynamic_truncate:
        _max_seq_len = opt.max_seq_len - len(aspect.split(' '))
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

    tokenizer.bos_token = tokenizer.bos_token if tokenizer.bos_token else '[CLS]'
    tokenizer.eos_token = tokenizer.eos_token if tokenizer.eos_token else '[SEP]'
    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    text_raw = text_left + ' ' + aspect + ' ' + text_right
    text_spc = bos_token + ' ' + text_raw + ' ' + eos_token + ' ' + aspect + ' ' + eos_token
    text_bert_indices = text_to_sequence(tokenizer, text_spc, opt.max_seq_len)
    text_raw_bert_indices = text_to_sequence(tokenizer, bos_token + ' ' + text_raw + ' ' + eos_token, opt.max_seq_len)
    aspect_bert_indices = text_to_sequence(tokenizer, aspect, opt.max_seq_len)

    aspect_begin = len(tokenizer.tokenize(bos_token + ' ' + text_left))
    aspect_position = set(range(aspect_begin, aspect_begin + np.count_nonzero(aspect_bert_indices)))

    # if 'lcfs' in opt.model_name or 'ssw_s' in opt.model_name or opt.use_syntax_based_SRD:
    #     syntactical_dist, _ = get_syntax_distance(text_raw, aspect, tokenizer, opt)
    # else:
    #     syntactical_dist = None

    if 'lcfs_vec' in input_demands:
        syntactical_dist, _ = get_syntax_distance(text_raw, aspect, tokenizer, opt)
        if opt.lcf == 'cdm':
            lcfs_vec = get_lca_ids_and_cdm_vec(opt, text_bert_indices, aspect_bert_indices, aspect_begin, syntactical_dist)
        elif opt.lcf == 'cdw':
            lcfs_vec = get_cdw_vec(opt, text_bert_indices, aspect_bert_indices, aspect_begin, syntactical_dist)
    if 'lcf_vec' in input_demands:
        if opt.lcf == 'cdm':
            lcf_vec = get_lca_ids_and_cdm_vec(opt, text_bert_indices, aspect_bert_indices, aspect_begin, None)
        elif opt.lcf == 'cdw':
            lcf_vec = get_cdw_vec(opt, text_bert_indices, aspect_bert_indices, aspect_begin, None)

    inputs = {
        'text_raw': text_raw,
        'text_spc': text_spc,
        'aspect': aspect,
        'aspect_position': aspect_position,
        'text_bert_indices': text_bert_indices,
        'text_raw_bert_indices': text_raw_bert_indices,
        'aspect_bert_indices': aspect_bert_indices,
        'lcf_vec': lcf_vec if 'lcf_vec' in input_demands else 0,
        'lcfs_vec': lcfs_vec if 'lcfs_vec' in input_demands else 0,
    }

    return inputs


def text_to_sequence(tokenizer, text, max_seq_len):
    return pad_and_truncate(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)), max_seq_len)


def get_syntax_distance(text_raw, aspect, tokenizer, opt):
    # Find distance in dependency parsing tree
    if isinstance(text_raw, list):
        text_raw = ' '.join(text_raw)

    if isinstance(aspect, list):
        aspect = ' '.join(aspect)

    try:
        raw_tokens, dist, max_dist = calculate_dep_dist(text_raw, aspect)
    except Exception as e:
        print('Text: {} Aspect: {}'.format(text_raw, aspect))
        raise RuntimeError('Ignore failure in calculate the syntax based SRD: {}, maybe the aspect is None'.format(e))

    if opt.model_name == 'dlcf_dca_bert':
        dist.insert(0, 0)
        dist.append(0)
    else:
        dist.insert(0, max(dist))
        dist.append(max(dist))
    raw_tokens.insert(0, tokenizer.bos_token)
    raw_tokens.append(tokenizer.eos_token)

    # the following two functions are both designed to calculate syntax-based distances
    if opt.srd_alignment:
        syntactical_dist = syntax_distance_alignment(raw_tokens, dist, opt.max_seq_len, tokenizer)
    else:
        syntactical_dist = pad_syntax_based_srd(raw_tokens, dist, tokenizer, opt)[1]
    return syntactical_dist, max_dist


def get_lca_ids_and_cdm_vec(opt, bert_spc_indices, aspect_indices, aspect_begin, syntactical_dist=None):
    SRD = opt.SRD
    cdm_vec = np.zeros((opt.max_seq_len), dtype=np.int64)
    aspect_len = np.count_nonzero(aspect_indices)
    text_len = np.count_nonzero(bert_spc_indices) - np.count_nonzero(aspect_indices) - 1
    if syntactical_dist is not None:
        for i in range(min(text_len, opt.max_seq_len)):
            if syntactical_dist[i] <= SRD:
                cdm_vec[i] = 1
    else:
        local_context_begin = max(0, aspect_begin - SRD)
        local_context_end = min(aspect_begin + aspect_len + SRD - 1, opt.max_seq_len)
        for i in range(min(text_len, opt.max_seq_len)):
            if local_context_begin <= i <= local_context_end:
                cdm_vec[i] = 1
    return cdm_vec


def get_cdw_vec(opt, bert_spc_indices, aspect_indices, aspect_begin, syntactical_dist=None):
    SRD = opt.SRD
    cdw_vec = np.zeros((opt.max_seq_len), dtype=np.float32)
    aspect_len = np.count_nonzero(aspect_indices)
    text_len = np.count_nonzero(bert_spc_indices) - np.count_nonzero(aspect_indices) - 1
    if syntactical_dist is not None:
        for i in range(min(text_len, opt.max_seq_len)):
            if syntactical_dist[i] > SRD:
                w = 1 - syntactical_dist[i] / text_len
                cdw_vec[i] = w
            else:
                cdw_vec[i] = 1
    else:
        local_context_begin = max(0, aspect_begin - SRD)
        local_context_end = min(aspect_begin + aspect_len + SRD - 1, opt.max_seq_len)
        for i in range(min(text_len, opt.max_seq_len)):
            if i < local_context_begin:
                w = 1 - (local_context_begin - i) / text_len
            elif local_context_begin <= i <= local_context_end:
                w = 1
            else:
                w = 1 - (i - local_context_end) / text_len
            try:
                assert 0 <= w <= 1  # exception
            except:
                pass
                # print('Warning! invalid CDW weight:', w)
            cdw_vec[i] = w
    return cdw_vec


def build_spc_mask_vec(opt, text_ids):
    spc_mask_vec = np.zeros((opt.max_seq_len, opt.hidden_dim), dtype=np.float32)
    for i in range(len(text_ids)):
        spc_mask_vec[i] = np.ones((opt.hidden_dim), dtype=np.float32)
    return spc_mask_vec


def build_sentiment_window(examples, tokenizer, similarity_threshold):
    copy_side_aspect('left', examples[0], examples[0], examples)
    for idx in range(1, len(examples)):
        if is_similar(examples[idx - 1]['text_bert_indices'],
                      examples[idx]['text_bert_indices'],
                      tokenizer=tokenizer,
                      similarity_threshold=similarity_threshold):
            copy_side_aspect('right', examples[idx - 1], examples[idx], examples)
            copy_side_aspect('left', examples[idx], examples[idx - 1], examples)
        else:
            copy_side_aspect('right', examples[idx - 1], examples[idx - 1], examples)
            copy_side_aspect('left', examples[idx], examples[idx], examples)
    copy_side_aspect('right', examples[-1], examples[-1], examples)
    return examples


def copy_side_aspect(direct, target, source, examples):
    if 'cluster_ids' not in target:
        target['cluster_ids'] = copy.deepcopy(target['aspect_position'])
        target['side_ex_ids'] = copy.deepcopy(set([target['ex_id']]))
    if 'cluster_ids' not in source:
        source['cluster_ids'] = copy.deepcopy(source['aspect_position'])
        source['side_ex_ids'] = copy.deepcopy(set([source['ex_id']]))

    if target['polarity'] == source['polarity']:

        target['side_ex_ids'] |= source['side_ex_ids']
        source['side_ex_ids'] |= target['side_ex_ids']
        target['cluster_ids'] |= source['cluster_ids']
        source['cluster_ids'] |= target['cluster_ids']

        for ex_id in target['side_ex_ids']:
            examples[ex_id]['cluster_ids'] |= source['cluster_ids']
            examples[ex_id]['side_ex_ids'] |= target['side_ex_ids']

    for data_item in ['text_bert_indices', 'lcf_vec', 'lcfs_vec']:
        target[direct + '_' + data_item] = source[data_item]
    target[direct + '_dist'] = int(abs(np.average(list(source['aspect_position'])) - np.average(list(target['aspect_position']))))
    # target[direct + '_dist'] = 0 if id(source['lcf_vec']) == id(target['lcf_vec']) else 1


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


def configure_spacy_model(opt):
    if not hasattr(opt, 'spacy_model'):
        opt.spacy_model = 'en_core_web_sm'
    global nlp
    try:
        nlp = spacy.load(opt.spacy_model)
    except:
        print('Can not load {} from spacy, try to download it in order to parse syntax tree:'.format(opt.spacy_model),
              termcolor.colored('\npython -m spacy download {}'.format(opt.spacy_model), 'green'))
        try:
            os.system('python -m spacy download {}'.format(opt.spacy_model))
            nlp = spacy.load(opt.spacy_model)
        except:
            raise RuntimeError('Download failed, you can download {} manually.'.format(opt.spacy_model))
    return nlp


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
    max_dist_temp = []
    for i, word in enumerate(doc):
        source = '{}_{}'.format(word.lower_, word.i)
        sum = 0
        flag = 1
        max_dist = 0
        for term_id, term in zip(term_ids, terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph, source=source, target=target)
            except:
                sum += len(doc)  # No connection between source and target
                flag = 0
        dist[i] = sum / len(terms)
        text[i] = word.text
        if flag == 1:
            max_dist_temp.append(sum / len(terms))
        if dist[i] > max_dist:
            max_dist = dist[i]
    return text, dist, max_dist
