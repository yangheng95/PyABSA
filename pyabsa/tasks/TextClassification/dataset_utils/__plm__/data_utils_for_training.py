# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels


class BERTTCDataset(PyABSADataset):
    bert_baseline_input_colses = {
        'bert_mlp': ['text_bert_indices'],
    }

    def __init__(self, config, tokenizer, dataset_type='train'):
        self.config = config
        lines = load_dataset_from_file(self.config.dataset_file[dataset_type])

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split('$LABEL$')
            text, label = line[0], line[1]
            text = text.strip()
            label = label.strip()
            text_indices = tokenizer.text_to_sequence(
                '{} {} {}'.format(tokenizer.tokenizer.cls_token, text, tokenizer.tokenizer.sep_token))

            data = {
                'text_bert_indices': text_indices,
                'label': label,
            }

            label_set.add(label)

            all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)

        self.data = all_data

        super().__init__(config)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
