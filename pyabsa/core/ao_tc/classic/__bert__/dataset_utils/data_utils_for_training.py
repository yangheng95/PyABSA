# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle

import numpy as np
import tqdm
from findfile import find_file
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, TransformerConnectionError


class Tokenizer4Pretraining:
    def __init__(self, max_seq_len, opt):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert, do_lower_case='uncased' in opt.pretrained_bert)
        except ValueError as e:
            raise TransformerConnectionError()

        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return self.tokenizer.encode(text, truncation=True, padding='max_length', max_length=self.max_seq_len, return_tensors='pt')


class AOBERTTCDataset(Dataset):
    bert_baseline_input_colses = {
        'aobert': ['text_bert_indices']
    }

    def __init__(self, dataset_list, tokenizer, opt):
        lines = load_apc_datasets(dataset_list)

        all_data = []

        label_set1 = set()
        label_set2 = set()
        label_set3 = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='building word indices...'):
            line = lines[i].strip().split('$LABEL$')
            text, labels = line[0], line[1]
            text = text.strip()
            label, advdet_label, ood_label = labels.strip().split(',')
            label, advdet_label, ood_label = label.strip(), advdet_label.strip(), ood_label.strip()
            text_indices = tokenizer.text_to_sequence('{}'.format(text))

            data = {
                'text_bert_indices': text_indices[0],

                'text_raw': text,

                'label': label,

                'advdet_label': advdet_label,

                'ood_label': ood_label,
            }

            label_set1.add(label)
            label_set2.add(advdet_label)
            label_set3.add(ood_label)

            all_data.append(data)

        check_and_fix_labels(label_set1, 'label', all_data, opt)
        check_and_fix_adv_labels(label_set2, 'advdet_label', all_data, opt)
        check_and_fix_ood_labels(label_set3, 'ood_label', all_data, opt)
        opt.polarities_dim1 = len(label_set1)
        opt.polarities_dim2 = len(label_set2)
        opt.polarities_dim3 = len(label_set3)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def check_and_fix_adv_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:
        adv_label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_adv_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        adv_label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_adv_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_adv_label' not in opt.args:
        opt.index_to_adv_label = index_to_adv_label
        opt.adv_label_to_index = adv_label_to_index

    if opt.index_to_adv_label != index_to_adv_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_adv_label.update(index_to_adv_label)
        opt.adv_label_to_index.update(adv_label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = adv_label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = adv_label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))


def check_and_fix_ood_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:
        ood_label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_ood_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        ood_label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_ood_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_ood_label' not in opt.args:
        opt.index_to_ood_label = index_to_ood_label
        opt.ood_label_to_index = ood_label_to_index

    if opt.index_to_ood_label != index_to_ood_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_ood_label.update(index_to_ood_label)
        opt.ood_label_to_index.update(ood_label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = ood_label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = ood_label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))
