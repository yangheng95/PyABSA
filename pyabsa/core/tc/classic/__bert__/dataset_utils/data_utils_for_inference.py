# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import tqdm
from findfile import find_cwd_dir
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.core.apc.dataset_utils.apc_utils import load_apc_datasets, LABEL_PADDING


class Tokenizer4Pretraining:
    def __init__(self, max_seq_len, opt, **kwargs):
        if kwargs.get('offline', False):
            self.tokenizer = AutoTokenizer.from_pretrained(find_cwd_dir(opt.pretrained_bert.split('/')[-1]),
                                                           do_lower_case='uncased' in opt.pretrained_bert)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert,
                                                           do_lower_case='uncased' in opt.pretrained_bert)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        # sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        # if len(sequence) == 0:
        #     sequence = [0]
        # if reverse:
        #     sequence = sequence[::-1]
        # return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return self.tokenizer.encode(text, truncation=True, padding='max_length', max_length=self.max_seq_len,
                                     return_tensors='pt')


class BERTClassificationDataset(Dataset):

    def __init__(self, tokenizer, opt):
        self.bert_baseline_input_colses = {
            'bert': ['text_bert_indices']
        }

        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_apc_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, postfix='preparing text classification dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                if '!ref!' in text:
                    text, _, label = text.strip().partition('!ref!')
                    label = label.strip()
                    text = text.strip().lower()

                else:
                    label = LABEL_PADDING

                text_indices = self.tokenizer.text_to_sequence('{}'.format(text))

                data = {
                    'ex_id': ex_id,

                    'text_bert_indices': text_indices[0]
                    if 'text_bert_indices' in self.opt.model.inputs else 0,

                    'text_raw': text,

                    'label': label,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text)
                else:
                    raise e

        self.all_data = all_data
        return self.all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)
