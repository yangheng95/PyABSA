# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import torch
import tqdm

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels


class GloVeTCDataset(PyABSADataset):

    def load_data_from_dict(self, data):
        pass

    def load_data_from_file(self, file_path):
        pass

    glove_input_colses = {
        'lstm': ['text_indices']
    }

    def __init__(self, config, tokenizer, dataset_type='train'):
        self.config = config
        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split('$LABEL$')
            text, label = line[0], line[1]
            text = text.strip().lower()
            label = label.strip().lower()
            text_indices = tokenizer.text_to_sequence(text)

            data = {
                'text_indices': text_indices,

                'label': label,
            }

            label_set.add(label)

            all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, config)
        config.output_dim = len(label_set)

        self.data = all_data

        super().__init__(config)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
