# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .apc_utils import copy_side_aspect, is_similar
from .apc_utils import get_lca_ids_and_cdm_vec, get_cdw_vec
from .apc_utils import get_syntax_distance, build_spc_mask_vec
from .apc_utils import load_datasets, prepare_input_from_text

from .apc_utils import SENTIMENT_PADDING


class ABSADataset(Dataset):

    def __init__(self, tokenizer, opt):
        self.input_colses = {
            'bert_base': ['text_raw_bert_indices'],
            'bert_spc': ['text_raw_bert_indices'],
            'lca_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lca_ids', 'lcf_vec'],
            'lcf_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec'],
            'slide_lcf_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
            'slide_lcfs_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
            'lcfs_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec'],
        }
        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        _text = text
        samples = []
        try:
            if '!sent!' not in text:
                splits = text.split('[ASP]')
                for i in range(0, len(splits) - 1, 2):
                    sample = text.replace('[ASP]', '').replace(splits[i + 1], '[ASP]' + splits[i + 1] + '[ASP]')
                    samples.append(sample)
            else:
                text, ref_sent = text.split('!sent!')
                ref_sent = ref_sent.split()
                text = '[PADDING] ' + text + ' [PADDING]'
                splits = text.split('[ASP]')

                if int((len(splits) - 1) / 2) == len(ref_sent):
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        sample += ' !sent! ' + str(ref_sent[int(i / 2)])
                        samples.append(sample.replace('[TEMP]', '[ASP]'))
                else:
                    print(_text,
                          ' -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.')
                    for i in range(0, len(splits) - 1, 2):
                        sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                              '[TEMP]' + splits[i + 1] + '[TEMP]').replace('[ASP]', '')
                        samples.append(sample.replace('[TEMP]', '[ASP]'))

        except:
            print('Invalid Input:', _text)
        return samples

    def prepare_infer_sample(self, text: str):
        self.process_data(self.parse_sample(text))

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []

        for text in tqdm(samples, postfix='building word indices...'):
            try:
                # handle for empty lines in inferring dataset
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '!sent!' in text:
                    text, polarity = text.split('!sent!')[0].strip(), text.split('!sent!')[1].strip()
                    polarity = int(polarity) if polarity else SENTIMENT_PADDING
                    if polarity < 0:
                        raise RuntimeError(
                            'Invalid sentiment label detected, only please label the sentiment between {0, N-1} '
                            '(assume there are N types of sentiment polarities.)')
                else:
                    polarity = SENTIMENT_PADDING

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split('[ASP]')
                text_left = text_left.replace('[PADDING] ', '')
                text_right = text_right.replace(' [PADDING]', '')

                # dynamic truncation on input text
                text_left = ' '.join(
                    text_left.split(' ')[int(-(self.tokenizer.max_seq_len - len(aspect.split())) / 2) - 1:])
                text_right = ' '.join(
                    text_right.split(' ')[:int((self.tokenizer.max_seq_len - len(aspect.split())) / 2) + 1])

                prepared_inputs = prepare_input_from_text(self.opt,
                                                          self.tokenizer,
                                                          text_left,
                                                          text_right,
                                                          aspect
                                                          )

                text_raw = prepared_inputs['text_raw']
                text_spc = prepared_inputs['text_spc']
                aspect = prepared_inputs['aspect']
                text_bert_indices = prepared_inputs['text_bert_indices']
                text_raw_bert_indices = prepared_inputs['text_raw_bert_indices']
                aspect_bert_indices = prepared_inputs['aspect_bert_indices']
                syntactical_dist = get_syntax_distance(text_raw, aspect, self.tokenizer) \
                    if 'lcfs' in self.opt.model_name else None

                if 'lca' in self.opt.model_name:
                    lca_ids, lcf_vec = get_lca_ids_and_cdm_vec(self.opt, text_bert_indices,
                                                               aspect_bert_indices,
                                                               syntactical_dist)
                    lcf_vec = torch.from_numpy(lcf_vec)
                    lca_ids = torch.from_numpy(lca_ids).long()
                elif 'lcf' in self.opt.model_name:
                    if 'cdm' in self.opt.lcf:
                        _, lcf_vec = get_lca_ids_and_cdm_vec(self.opt, text_bert_indices,
                                                             aspect_bert_indices,
                                                             syntactical_dist)
                        lcf_vec = torch.from_numpy(lcf_vec)
                    elif 'cdw' in self.opt.lcf:
                        lcf_vec = get_cdw_vec(self.opt, text_bert_indices,
                                              aspect_bert_indices,
                                              syntactical_dist)
                        lcf_vec = torch.from_numpy(lcf_vec)
                    elif 'fusion' in self.opt.lcf:
                        raise NotImplementedError('LCF-Fusion is not recommended due to its low efficiency!')
                    else:
                        raise KeyError('Invalid LCF Mode!')

                data = {
                    'text_raw': text_raw,
                    'aspect': aspect,
                    'lca_ids': lca_ids if 'lca_ids' in self.input_colses[self.opt.model_name] else 0,
                    'lcf_vec': lcf_vec if 'lcf_vec' in self.input_colses[self.opt.model_name] else 0,
                    'spc_mask_vec': build_spc_mask_vec(self.opt, text_raw_bert_indices)
                    if 'spc_mask_vec' in self.input_colses[self.opt.model_name] else 0,
                    'text_bert_indices': text_bert_indices if 'text_bert_indices' in self.input_colses[
                        self.opt.model_name] else 0,
                    'aspect_bert_indices': aspect_bert_indices if 'aspect_bert_indices' in self.input_colses[
                        self.opt.model_name] else 0,
                    'text_raw_bert_indices': text_raw_bert_indices
                    if 'text_raw_bert_indices' in self.input_colses[self.opt.model_name] else 0,
                    'polarity': polarity,
                }

                for _, item in enumerate(data):
                    data[item] = torch.tensor(data[item]) if type(item) is not str else data[item]
                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text)
                else:
                    raise e

        if all_data and 'slide' in self.opt.model_name:
            copy_side_aspect('left', all_data[0], all_data[0])
            for idx in range(1, len(all_data)):
                if is_similar(all_data[idx - 1]['text_bert_indices'],
                              all_data[idx]['text_bert_indices']):
                    copy_side_aspect('right', all_data[idx - 1], all_data[idx])
                    copy_side_aspect('left', all_data[idx], all_data[idx - 1])
                else:
                    copy_side_aspect('right', all_data[idx - 1], all_data[idx - 1])
                    copy_side_aspect('left', all_data[idx], all_data[idx])
            copy_side_aspect('right', all_data[-1], all_data[-1])
        self.all_data = all_data
        return all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)
