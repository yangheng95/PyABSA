# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 2021/5/31 0031
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
import tqdm
from torch.utils.data import Dataset

from pyabsa.utils.pyabsa_utils import check_and_fix_labels
from .apc_utils import build_sentiment_window, build_spc_mask_vec, load_apc_datasets, prepare_input_for_apc
from .apc_utils_for_dlcf_dca import prepare_input_for_dlcf_dca


class ABSADataset(Dataset):

    def __init__(self, fname, tokenizer, opt):
        ABSADataset.opt = opt

        lines = load_apc_datasets(fname)

        all_data = []
        # record polarities type to update polarities_dim
        label_set = set()

        for i in tqdm.tqdm(range(0, len(lines), 3), postfix='building word indices...'):
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            polarity = int(polarity)
            label_set.add(polarity)

            prepared_inputs = prepare_input_for_apc(opt, tokenizer, text_left, text_right, aspect)

            text_raw = prepared_inputs['text_raw']
            aspect = prepared_inputs['aspect']
            aspect_position = prepared_inputs['aspect_position']
            text_bert_indices = prepared_inputs['text_bert_indices']
            text_raw_bert_indices = prepared_inputs['text_raw_bert_indices']
            aspect_bert_indices = prepared_inputs['aspect_bert_indices']
            lcf_vec = prepared_inputs['lcf_cdm_vec'] if opt.lcf == 'cdm' else prepared_inputs['lcf_cdw_vec']
            if opt.model_name == 'dlcf_dca_bert':
                prepared_inputs = prepare_input_for_dlcf_dca(opt, tokenizer, text_left, text_right, aspect)
                dlcf_vec = prepared_inputs['dlcf_cdm_vec'] if opt.lcf == 'cdm' else prepared_inputs['dlcf_cdw_vec']
                depend_ids = prepared_inputs['depend_ids']
                depended_ids = prepared_inputs['depended_ids']
                no_connect = prepared_inputs['no_connect']
            data = {
                'ex_id': i // 3,

                'text_raw': text_raw,

                'aspect': aspect,

                'aspect_position': aspect_position,

                'lcf_vec': lcf_vec if 'lcf_vec' in opt.model.inputs else 0,

                'dlcf_vec': dlcf_vec if 'dlcf_vec' in opt.model.inputs else 0,

                'spc_mask_vec': build_spc_mask_vec(opt, text_raw_bert_indices)
                if 'spc_mask_vec' in opt.model.inputs else 0,

                'text_bert_indices': text_bert_indices
                if 'text_bert_indices' in opt.model.inputs else 0,

                'aspect_bert_indices': aspect_bert_indices
                if 'aspect_bert_indices' in opt.model.inputs else 0,

                'text_raw_bert_indices': text_raw_bert_indices
                if 'text_raw_bert_indices' in opt.model.inputs else 0,

                'depend_ids': depend_ids if 'depend_ids' in opt.model.inputs else 0,

                'depended_ids': depended_ids if 'depended_ids' in opt.model.inputs else 0,

                'no_connect': no_connect if 'no_connect' in opt.model.inputs else 0,

                'polarity': polarity,
            }

            label_set.add(polarity)

            all_data.append(data)

        check_and_fix_labels(label_set, 'polarity', all_data)
        opt.polarities_dim = len(label_set)

        if opt.model_name in ['slide_lcf_bert', 'slide_lcfs_bert', 'ssw_t', 'ssw_s']:
            all_data = build_sentiment_window(all_data, tokenizer, opt.similarity_threshold)
            for data in all_data:

                cluster_ids = []
                for pad_idx in range(opt.max_seq_len):
                    if pad_idx in data['cluster_ids']:
                        cluster_ids.append(data['polarity'])
                    else:
                        cluster_ids.append(-100)
                        # cluster_ids.append(3)

                data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
                data['side_ex_ids'] = np.array(0)
                data['aspect_position'] = np.array(0)

        else:
            for data in all_data:
                data['aspect_position'] = np.array(0)

        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
