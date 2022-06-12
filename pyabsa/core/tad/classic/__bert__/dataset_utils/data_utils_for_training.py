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


class TADBERTTCDataset(Dataset):
    bert_baseline_input_colses = {
        'tadbert': ['text_bert_indices']
    }

    def __init__(self, dataset_list, tokenizer, opt):
        lines = load_apc_datasets(dataset_list)

        all_data = []

        label_set1 = set()
        label_set2 = set()
        label_set3 = set()

        for i in tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...'):
            line = lines[i].strip().split('$LABEL$')
            text, labels = line[0], line[1]
            text = text.strip()
            label, perturb_label, is_adv = labels.strip().split(',')
            label, perturb_label, is_adv = label.strip(), perturb_label.strip(), is_adv.strip()

            # if label == perturb_label:
            #     continue

            if is_adv == '1' or is_adv == 1:
                label = '-100'

            text_indices = tokenizer.text_to_sequence('{}'.format(text))

            data = {
                'text_bert_indices': text_indices[0],

                'text_raw': text,

                'label': label,

                'perturb_label': perturb_label,

                'is_adv': is_adv,
            }

            label_set1.add(label)
            label_set2.add(perturb_label)
            label_set3.add(is_adv)

            all_data.append(data)

        check_and_fix_labels(label_set1, 'label', all_data, opt)
        check_and_fix_perturb_labels(label_set2, 'perturb_label', all_data, opt)
        check_and_fix_is_adv_labels(label_set3, 'is_adv', all_data, opt)
        opt.class_dim = len(label_set1-{'-100'})
        opt.adv_det_dim = len(label_set3-{'-100'})

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def check_and_fix_perturb_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:
        perturb_label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_perturb_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        perturb_label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_perturb_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_perturb_label' not in opt.args:
        opt.index_to_perturb_label = index_to_perturb_label
        opt.perturb_label_to_index = perturb_label_to_index

    if opt.index_to_perturb_label != index_to_perturb_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_perturb_label.update(index_to_perturb_label)
        opt.perturb_label_to_index.update(perturb_label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = perturb_label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = perturb_label_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))


def check_and_fix_is_adv_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:
        is_adv_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_is_adv = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        is_adv_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_is_adv = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_is_adv' not in opt.args:
        opt.index_to_is_adv = index_to_is_adv
        opt.is_adv_to_index = is_adv_to_index

    if opt.index_to_is_adv != index_to_is_adv:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_is_adv.update(index_to_is_adv)
        opt.is_adv_to_index.update(is_adv_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = is_adv_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = is_adv_to_index[item.polarity]
    print('Dataset Label Details: {}'.format(num_label))
