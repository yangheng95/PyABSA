# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/31 0031
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import tqdm
from torch.utils.data import Dataset
from .apc_utils import is_similar, copy_side_aspect
from .apc_utils import get_lca_ids_and_cdm_vec, get_cdw_vec, get_syntax_distance, build_spc_mask_vec


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

        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        all_data = []

        for i in tqdm.tqdm(range(0, len(lines), 3), postfix='building word indices...'):

            text_left, _, text_right = [s.strip().lower() for s in " ".join(lines[i].split()).partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity) + 1

            # # dynamic truncation on input text
            # text_left = ' '.join(
            #     text_left.split(' ')[int(-(tokenizer.max_seq_len - len(aspect.split())) / 2) - 1:])
            # text_right = ' '.join(
            #     text_right.split(' ')[:int((tokenizer.max_seq_len - len(aspect.split())) / 2) + 1])

            text_raw = text_left + ' ' + aspect + ' ' + text_right
            text_spc = '[CLS] ' + text_raw + ' [SEP] ' + aspect + ' [SEP]'
            text_bert_indices = tokenizer.text_to_sequence(text_spc)
            text_raw_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_raw + ' [SEP]')
            aspect_bert_indices = tokenizer.text_to_sequence('[CLS] ' + aspect + ' [SEP]')

            syntactical_dist = get_syntax_distance(text_raw, aspect, tokenizer)

            if 'lca' in opt.model_name:
                lca_ids, lcf_vec = get_lca_ids_and_cdm_vec(opt,
                                                           text_bert_indices,
                                                           aspect_bert_indices,
                                                           syntactical_dist)
            elif 'lcf' in opt.model_name:
                if 'cdm' in opt.lcf:
                    _, lcf_vec = get_lca_ids_and_cdm_vec(opt, text_bert_indices, aspect_bert_indices, syntactical_dist)
                elif 'cdw' in opt.lcf:
                    lcf_vec = get_cdw_vec(opt, text_bert_indices, aspect_bert_indices, syntactical_dist)
                elif 'fusion' in opt.lcf:
                    raise NotImplementedError('LCF-Fusion is not recommended due to its low efficiency!')
                else:
                    raise KeyError('Invalid LCF Mode!')

            data = {
                'text_raw': text_raw,

                'aspect': aspect,

                'lca_ids': lca_ids if 'lca_ids' in ABSADataset.input_colses[opt.model_name] else 0,

                'lcf_vec': lcf_vec if 'lcf_vec' in ABSADataset.input_colses[opt.model_name] else 0,

                'spc_mask_vec': build_spc_mask_vec(opt, text_raw_bert_indices)
                if 'spc_mask_vec' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_bert_indices': text_bert_indices
                if 'text_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'aspect_bert_indices': aspect_bert_indices
                if 'aspect_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'text_raw_bert_indices': text_raw_bert_indices
                if 'text_raw_bert_indices' in ABSADataset.input_colses[opt.model_name] else 0,

                'polarity': polarity,
            }

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
