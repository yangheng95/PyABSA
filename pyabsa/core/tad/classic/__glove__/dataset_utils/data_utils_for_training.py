# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import tqdm
from torch.utils.data import Dataset

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, pad_and_truncate


def build_tokenizer(dataset_list, max_seq_len, dat_fname, opt):
    dataset_name = os.path.basename(opt.dataset_name)
    if not os.path.exists('run/{}'.format(dataset_name)):
        os.makedirs('run/{}'.format(dataset_name))
    tokenizer_path = 'run/{}/{}'.format(dataset_name, dat_fname)
    if os.path.exists(tokenizer_path):
        print('Loading tokenizer on {}'.format(tokenizer_path))
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    else:
        text = ''
        for dataset_type in dataset_list:
            for file in dataset_list[dataset_type]:
                fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
                lines = fin.readlines()
                fin.close()
                for i in range(0, len(lines)):
                    text += lines[i].lower().strip()

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
    return tokenizer


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class GloVeTADDataset(Dataset):
    glove_input_colses = {
        'tadlstm': ['text_indices']
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
            label, is_adv, adv_train_label = labels.strip().split(',')
            label, is_adv, adv_train_label = label.strip(), is_adv.strip(), adv_train_label.strip()

            if is_adv == '1' or is_adv == 1:
                adv_train_label = label
                label = '-100'
            else:
                label = label
                adv_train_label = '-100'

            text_indices = tokenizer.text_to_sequence('{}'.format(text))

            data = {
                'text_indices': text_indices,

                'text_raw': text,

                'label': label,

                'adv_train_label': adv_train_label,

                'is_adv': is_adv,
            }

            label_set1.add(label)
            label_set2.add(adv_train_label)
            label_set3.add(is_adv)

            all_data.append(data)

        check_and_fix_labels(label_set1, 'label', all_data, opt)
        check_and_fix_adv_train_labels(label_set2, 'adv_train_label', all_data, opt)
        check_and_fix_is_adv_labels(label_set3, 'is_adv', all_data, opt)
        opt.class_dim = len(label_set1 - {'-100'})
        opt.adv_det_dim = len(label_set3 - {'-100'})

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def check_and_fix_adv_train_labels(label_set: set, label_name, all_data, opt):
    # update polarities_dim, init model behind execution of this function!
    if '-100' in label_set:
        adv_train_label_to_index = {origin_label: int(idx) - 1 if origin_label != '-100' else -100 for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_adv_train_label = {int(idx) - 1 if origin_label != '-100' else -100: origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    else:
        adv_train_label_to_index = {origin_label: int(idx) for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
        index_to_adv_train_label = {int(idx): origin_label for origin_label, idx in zip(sorted(label_set), range(len(label_set)))}
    if 'index_to_adv_train_label' not in opt.args:
        opt.index_to_adv_train_label = index_to_adv_train_label
        opt.adv_train_label_to_index = adv_train_label_to_index

    if opt.index_to_adv_train_label != index_to_adv_train_label:
        # raise KeyError('Fail to fix the labels, the number of labels are not equal among all datasets!')
        opt.index_to_adv_train_label.update(index_to_adv_train_label)
        opt.adv_train_label_to_index.update(adv_train_label_to_index)
    num_label = {l: 0 for l in label_set}
    num_label['Sum'] = len(all_data)
    for item in all_data:
        try:
            num_label[item[label_name]] += 1
            item[label_name] = adv_train_label_to_index[item[label_name]]
        except Exception as e:
            # print(e)
            num_label[item.polarity] += 1
            item.polarity = adv_train_label_to_index[item.polarity]
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
