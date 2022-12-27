# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import tqdm
from termcolor import colored

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from ...dataset_utils.__plm__.classic_bert_apc_utils import prepare_input_for_apc, build_sentiment_window
from ...dataset_utils.__plm__.dependency_graph import configure_spacy_model, prepare_dependency_graph
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, validate_example, fprint


class BERTBaselineABSADataset(PyABSADataset):

    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):

        configure_spacy_model(self.config)

        lines = load_dataset_from_file(self.config.dataset_file[self.dataset_type], config=self.config)

        all_data = []
        label_set = set()

        dep_cache_path = os.path.join(os.getcwd(), 'run/{}/dependency_cache/'.format(self.config.dataset_name))
        if not os.path.exists(dep_cache_path):
            os.makedirs(dep_cache_path)
        graph_path = prepare_dependency_graph(self.config.dataset_file[self.dataset_type], dep_cache_path,
                                              self.config.max_seq_len, self.config)
        fin = open(graph_path, 'rb')
        idx2graph = pickle.load(fin)

        ex_id = 0

        if len(lines) % 3 != 0:
            fprint(colored(
                'ERROR: one or more datasets are corrupted, make sure the number of lines in a dataset should be multiples of 3.',
                'red'))

        for i in tqdm.tqdm(range(0, len(lines), 3), desc='preparing dataloader'):
            if lines[i].count("$T$") > 1:
                continue
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text_raw = text_left + ' ' + aspect + ' ' + text_right
            polarity = lines[i + 2].strip()
            # polarity = int(polarity)

            if validate_example(text_raw, aspect, polarity, self.config):
                continue

            prepared_inputs = prepare_input_for_apc(self.config, self.tokenizer, text_left, text_right, aspect)

            aspect_position = prepared_inputs['aspect_position']

            # it is hard to decide whether [CLS] and [SEP] should be added into sequences, e.g., left_context or right_context,
            # so we disable all [CLS]s and [SEP]s
            text_indices = self.tokenizer.text_to_sequence(text_left + ' ' + aspect + ' ' + text_right)
            context_indices = self.tokenizer.text_to_sequence(text_left + text_right)
            left_indices = self.tokenizer.text_to_sequence(text_left)
            left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
            right_indices = self.tokenizer.text_to_sequence(text_right)
            right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + ' ' + text_right)

            aspect_indices = self.tokenizer.text_to_sequence(aspect)
            aspect_len = np.count_nonzero(aspect_indices)
            left_len = min(self.config.max_seq_len - aspect_len, np.count_nonzero(left_indices))
            left_indices = np.concatenate(
                (left_indices[:left_len], np.asarray([0] * (self.config.max_seq_len - left_len))))
            aspect_boundary = np.asarray([left_len, min(left_len + aspect_len - 1, self.config.max_seq_len)])

            dependency_graph = np.pad(idx2graph[text_raw],
                                      ((0, max(0, self.config.max_seq_len - idx2graph[text_raw].shape[0])),
                                       (0, max(0, self.config.max_seq_len - idx2graph[text_raw].shape[0]))),
                                      'constant')

            dependency_graph = dependency_graph[:, range(0, self.config.max_seq_len)]
            dependency_graph = dependency_graph[range(0, self.config.max_seq_len), :]

            data = {
                'ex_id': ex_id,

                'text_indices': text_indices
                if 'text_indices' in self.config.inputs_cols else 0,

                'context_indices': context_indices
                if 'context_indices' in self.config.inputs_cols else 0,

                'left_indices': left_indices
                if 'left_indices' in self.config.inputs_cols else 0,

                'left_with_aspect_indices': left_with_aspect_indices
                if 'left_with_aspect_indices' in self.config.inputs_cols else 0,

                'right_indices': right_indices
                if 'right_indices' in self.config.inputs_cols else 0,

                'right_with_aspect_indices': right_with_aspect_indices
                if 'right_with_aspect_indices' in self.config.inputs_cols else 0,

                'aspect_indices': aspect_indices
                if 'aspect_indices' in self.config.inputs_cols else 0,

                'aspect_boundary': aspect_boundary
                if 'aspect_boundary' in self.config.inputs_cols else 0,

                'aspect_position': aspect_position,

                'dependency_graph': dependency_graph
                if 'dependency_graph' in self.config.inputs_cols else 0,

                'polarity': polarity,
            }
            ex_id += 1

            label_set.add(polarity)

            all_data.append(data)

        check_and_fix_labels(label_set, 'polarity', all_data, self.config)
        self.config.output_dim = len(label_set)

        all_data = build_sentiment_window(all_data, self.tokenizer, self.config.similarity_threshold,
                                          input_demands=self.config.inputs_cols)
        for data in all_data:

            cluster_ids = []
            for pad_idx in range(self.config.max_seq_len):
                if pad_idx in data['cluster_ids']:
                    cluster_ids.append(data['polarity'])
                else:
                    cluster_ids.append(-100)
                    # cluster_ids.append(3)

            data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
            data['side_ex_ids'] = np.array(0)
            data['aspect_position'] = np.array(0)
        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type='train', **kwargs):
        super().__init__(config, tokenizer, dataset_type=dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
