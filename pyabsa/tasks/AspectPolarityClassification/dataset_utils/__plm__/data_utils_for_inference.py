# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import tqdm
from torch.utils.data import Dataset

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import validate_example
from .classic_bert_apc_utils import prepare_input_for_apc, build_sentiment_window
from .dependency_graph import dependency_adj_matrix, configure_spacy_model
from ..__lcf__.data_utils_for_inference import parse_sample


class BERTABSAInferenceDataset(Dataset):

    def __init__(self, config, tokenizer):
        configure_spacy_model(config)

        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def prepare_infer_sample(self, text: str, ignore_error=True):
        self.process_data(parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_dataset_from_file(infer_file, logger=self.config.logger)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []

        if len(samples) > 100:
            it = tqdm.tqdm(samples, postfix='preparing apc inference dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference dataset
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '$LABEL$' in text:
                    text, polarity = text.split('$LABEL$')[0].strip(), text.split('$LABEL$')[1].strip()
                    polarity = polarity if polarity else LabelPaddingOption.SENTIMENT_PADDING
                    text = text.replace('[PADDING]', '')

                else:
                    polarity = str(LabelPaddingOption.SENTIMENT_PADDING)

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split('[ASP]')
                text_left = text_left.replace('[PADDING] ', '').lower().strip()
                text_right = text_right.replace(' [PADDING]', '').lower().strip()
                aspect = aspect.lower().strip()
                text = text_left + ' ' + aspect + ' ' + text_right
                # polarity = int(polarity)

                if validate_example(text, aspect, polarity, self.config) or not aspect:
                    continue

                prepared_inputs = prepare_input_for_apc(self.config, self.tokenizer.tokenizer, text_left, text_right, aspect)

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
                left_indices = np.concatenate((left_indices[:left_len], np.asarray([0] * (self.config.max_seq_len - left_len))))
                aspect_boundary = np.asarray([left_len, min(left_len + aspect_len - 1, self.config.max_seq_len)])

                idx2graph = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right)
                dependency_graph = np.pad(idx2graph,
                                          ((0, max(0, self.config.max_seq_len - idx2graph.shape[0])),
                                           (0, max(0, self.config.max_seq_len - idx2graph.shape[0]))),
                                          'constant')

                dependency_graph = dependency_graph[:, range(0, self.config.max_seq_len)]
                dependency_graph = dependency_graph[range(0, self.config.max_seq_len), :]

                data = {
                    'ex_id': ex_id,

                    'text_bert_indices': text_indices
                    if 'text_bert_indices' in self.config.inputs_cols else 0,

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

                    'text_raw': text,
                    'aspect': aspect,
                    'polarity': polarity,

                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing: {} Error info:{}'.format(text, e))
                else:
                    raise RuntimeError('Catch Exception: {}, use ignore_error=True to remove error samples.'.format(e))

        self.data = all_data

        all_data = build_sentiment_window(all_data, self.tokenizer, self.config.similarity_threshold, input_demands=self.config.inputs_cols)
        for data in all_data:

            cluster_ids = []
            for pad_idx in range(self.config.max_seq_len):
                if pad_idx in data['cluster_ids']:
                    # print(data['polarity'])
                    cluster_ids.append(self.config.label_to_index.get(self.config.index_to_label.get(data['polarity'], 'N.A.'),
                                                                      LabelPaddingOption.SENTIMENT_PADDING))
                else:
                    cluster_ids.append(-100)
                    # cluster_ids.append(3)
            data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
            data['side_ex_ids'] = np.array(0)
            data['aspect_position'] = np.array(0)

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
