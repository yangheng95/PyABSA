# -*- coding: utf-8 -*-
# file: dependency_graph.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import os.path
import pickle

import numpy as np
import spacy
import termcolor
import tqdm
from spacy.tokens import Doc

from pyabsa.utils.pyabsa_utils import fprint


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def configure_spacy_model(config):
    if not hasattr(config, 'spacy_model'):
        config.spacy_model = 'en_core_web_sm'
    global nlp
    try:
        nlp = spacy.load(config.spacy_model)
    except:
        fprint('Can not load {} from spacy, try to download it in order to parse syntax tree:'.format(config.spacy_model),
               termcolor.colored('\npython -m spacy download {}'.format(config.spacy_model), 'green'))
        try:
            os.system('python -m spacy download {}'.format(config.spacy_model))
            nlp = spacy.load(config.spacy_model)
        except:
            raise RuntimeError('Download failed, you can download {} manually.'.format(config.spacy_model))

    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


def prepare_dependency_graph(dataset_list, graph_path, max_seq_len, opt):
    if 'train' in dataset_list[0].lower():
        append_name = 'train_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    elif 'test' in dataset_list[0].lower():
        append_name = 'test_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    elif 'val' in dataset_list[0].lower():
        append_name = 'val_set_{}x{}.graph'.format(max_seq_len, max_seq_len)
    else:
        append_name = 'unrecognized_set_{}x{}.graph'.format(max_seq_len, max_seq_len)

    graph_path = os.path.join(graph_path, append_name)

    if os.path.isfile(graph_path):
        return graph_path

    idx2graph = {}
    if os.path.isdir(graph_path):
        fout = open(os.path.join(graph_path, append_name), 'wb')
        graph_path = os.path.join(graph_path, append_name)
    elif os.path.isfile(graph_path):
        return graph_path
    else:
        fout = open(graph_path, 'wb')

    for filename in dataset_list:
        try:
            fprint('parsing dependency matrix:', filename)
            fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in tqdm.tqdm(range(0, len(lines), 3), postfix='Construct graph for {}'.format(filename)):
                text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].strip()
                adj_matrix = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
                text = text_left + ' ' + aspect + ' ' + text_right
                idx2graph[text.lower()] = adj_matrix
        except Exception as e:
            fprint(e)
            fprint('unprocessed:', filename)
    pickle.dump(idx2graph, fout)
    fout.close()
    return graph_path
