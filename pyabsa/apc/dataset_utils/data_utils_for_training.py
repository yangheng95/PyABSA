# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

# modified: yangheng<yangheng@m.scnu.edu.cn>


import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from .apc_utils import is_similar, copy_side_aspect, calculate_dep_dist


class ABSADataset(Dataset):
    input_colses = {
        'bert_base': ['text_raw_bert_indices'],
        'bert_spc': ['text_bert_indices'],
        'lca_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lca_ids', 'lcf_vec'],
        'lcf_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec'],
        'slide_lcf_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
        'slide_lcfs_bert': ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec'],
        'lcfs_bert': ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec'],
    }

    def __init__(self, fname, tokenizer, opt):

        ABSADataset.opt = opt

        def get_lca_ids_and_cdm_vec(text_ids, aspect_indices):
            SRD = opt.SRD
            lca_ids = np.zeros((opt.max_seq_len), dtype=np.float32)
            cdm_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            aspect_len = np.count_nonzero(aspect_indices) - 2
            # text_len = np.count_nonzero(text_ids)
            if 'lcfs' in opt.model_name:
                # Find distance in dependency parsing tree
                # raw_tokens, dist = calculate_dep_dist(text_spc, aspect)
                raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
                raw_tokens.insert(0, tokenizer.cls_token)
                dist.insert(0, 0)
                raw_tokens.append(tokenizer.sep_token)
                dist.append(0)
                syntactical_dist = tokenizer.tokenize(raw_tokens, dist)[1]
                for i in range(opt.max_seq_len):
                    if syntactical_dist[i] <= SRD:
                        lca_ids[i] = 1
                        cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            else:
                aspect_begin = get_asp_index(text_ids, aspect_indices)
                if aspect_begin < 0:
                    return lca_ids, cdm_vec
                local_context_begin = max(0, aspect_begin - SRD)
                local_context_end = aspect_begin + aspect_len + SRD - 1
                for i in range(opt.max_seq_len):
                    if local_context_begin <= i <= local_context_end:
                        lca_ids[i] = 1
                        cdm_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return lca_ids, cdm_vec

        def get_cdw_vec(text_ids, aspect_indices):
            SRD = opt.SRD
            cdw_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            aspect_len = np.count_nonzero(aspect_indices) - 2
            text_len = np.flatnonzero(text_ids)[-1] + 1
            if 'lcfs' in opt.model_name:
                # Find distance in dependency parsing tree
                raw_tokens, dist = calculate_dep_dist(text_raw, aspect)
                raw_tokens.insert(0, tokenizer.cls_token)
                dist.insert(0, 0)
                raw_tokens.append(tokenizer.sep_token)
                dist.append(0)
                syntactical_dist = tokenizer.tokenize(raw_tokens, dist)[1]
                for i in range(text_len):
                    if syntactical_dist[i] > SRD:
                        w = 1 - syntactical_dist[i] / text_len
                        # w = max(0, 1 - syntactical_dist[i] / text_len)
                        cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
                    else:
                        cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            else:
                aspect_begin = get_asp_index(text_ids, aspect_indices)
                if aspect_begin < 0:
                    return np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
                local_context_begin = max(0, aspect_begin - SRD)
                local_context_end = aspect_begin + aspect_len + SRD - 1
                for i in range(text_len):
                    if i < local_context_begin:
                        w = 1 - (local_context_begin - i) / text_len
                        cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)
                    elif local_context_begin <= i <= local_context_end:
                        cdw_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
                    elif i > local_context_end:
                        w = 1 - (i - local_context_end) / text_len
                        cdw_vec[i] = w * np.ones((opt.embed_dim), dtype=np.float32)

            return cdw_vec

        def get_asp_index(text_ids, aspect_indices):
            aspect_len = np.count_nonzero(aspect_indices) - 2
            aspect_indices = aspect_indices[1:aspect_len + 1]
            for i in range(len(text_ids)):
                for j in range(len(aspect_indices)):
                    if text_ids[i + j] == aspect_indices[j] and j == len(aspect_indices) - 1:
                        return i
                    elif text_ids[i + j] != aspect_indices[j]:
                        break

        def build_spc_mask_vec(text_ids):
            spc_mask_vec = np.zeros((opt.max_seq_len, opt.embed_dim), dtype=np.float32)
            for i in range(len(text_ids)):
                if text_ids[i] != 0:
                    spc_mask_vec[i] = np.ones((opt.embed_dim), dtype=np.float32)
            return spc_mask_vec

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []

        for i in tqdm.tqdm(range(0, len(lines), 3), postfix='building word indices...'):
            try:
                text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()
                polarity = int(polarity) + 1

                # dynamic truncation on input text
                text_left = ' '.join(
                    text_left.split(' ')[int(-(tokenizer.max_seq_len - len(aspect.split())) / 2) - 1:])
                text_right = ' '.join(
                    text_right.split(' ')[:int((tokenizer.max_seq_len - len(aspect.split())) / 2) + 1])

                text_raw = text_left + ' ' + aspect + ' ' + text_right
                text_spc = '[CLS] ' + text_raw + ' [SEP] ' + aspect + ' [SEP]'
                text_bert_indices = tokenizer.text_to_sequence(text_spc)
                text_raw_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_raw + ' [SEP]')
                aspect_bert_indices = tokenizer.text_to_sequence('[CLS] ' + aspect + ' [SEP]')

                if 'lca' in opt.model_name:
                    lca_ids, lcf_vec = get_lca_ids_and_cdm_vec(text_bert_indices, aspect_bert_indices)
                elif 'lcf' in opt.model_name:
                    if 'cdm' in opt.lcf:
                        _, lcf_vec = get_lca_ids_and_cdm_vec(text_bert_indices, aspect_bert_indices)
                    elif 'cdw' in opt.lcf:
                        lcf_vec = get_cdw_vec(text_bert_indices, aspect_bert_indices)
                    elif 'fusion' in opt.lcf:
                        raise NotImplementedError('LCF-Fusion is not recommended due to its low efficiency!')
                    else:
                        raise KeyError('Invalid LCF Mode!')

                data = {
                    'text_raw': text_raw,
                    'aspect': aspect,
                    'lca_ids': lca_ids if 'lca_ids' in ABSADataset.input_colses[opt.model_name] else 0,
                    'lcf_vec': lcf_vec if 'lcf_vec' in ABSADataset.input_colses[opt.model_name] else 0,
                    'spc_mask_vec': build_spc_mask_vec(text_raw_bert_indices) if 'spc_mask_vec' in
                                                                                 ABSADataset.input_colses[
                                                                                     opt.model_name] else 0,
                    'text_bert_indices': text_bert_indices if 'text_bert_indices' in ABSADataset.input_colses[
                        opt.model_name] else 0,
                    'aspect_bert_indices': aspect_bert_indices if 'aspect_bert_indices' in ABSADataset.input_colses[
                        opt.model_name] else 0,
                    'text_raw_bert_indices': text_raw_bert_indices
                    if 'text_raw_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                    'polarity': polarity,
                }
            except:
                print("Ignore preprocessing error on text:", text_raw)
                continue

            for _, item in enumerate(data):
                data[item] = torch.tensor(data[item]) if type(data[item]) is np.ndarray else data[item]
            all_data.append(data)

        if 'slide' in opt.model_name:
            copy_side_aspect('left', all_data[0], all_data[0])
            for idx in range(1, len(all_data)):
                if is_similar(all_data[idx - 1]['text_bert_indices'], all_data[idx]['text_bert_indices']):
                    copy_side_aspect('right', all_data[idx - 1], all_data[idx])
                    copy_side_aspect('left', all_data[idx], all_data[idx - 1])
                else:
                    copy_side_aspect('right', all_data[idx - 1], all_data[idx - 1])
                    copy_side_aspect('left', all_data[idx], all_data[idx])
            copy_side_aspect('right', all_data[-1], all_data[-1])

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
