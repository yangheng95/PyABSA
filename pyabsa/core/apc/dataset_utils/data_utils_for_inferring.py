# -*- coding: utf-8 -*-
# file: data_utils_for_inferring.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, validate_example
from torch.utils.data import Dataset
from tqdm import tqdm

from .apc_utils import (build_sentiment_window,
                        build_spc_mask_vec,
                        load_apc_datasets,
                        prepare_input_for_apc,
                        LABEL_PADDING, configure_spacy_model)
from .apc_utils_for_dlcf_dca import prepare_input_for_dlcf_dca, configure_dlcf_spacy_model


class ABSADataset(Dataset):

    def __init__(self, tokenizer, opt):
        configure_spacy_model(opt)
        self.tokenizer = tokenizer
        self.opt = opt
        self.all_data = []

    def parse_sample(self, text):
        _text = text
        samples = []

        if '!sent!' not in text:
            text += '!sent!'
        text, _, ref_sent = text.partition('!sent!')
        ref_sent = ref_sent.split(',') if ref_sent else None
        text = '[PADDING] ' + text + ' [PADDING]'
        splits = text.split('[ASP]')

        if ref_sent and int((len(splits) - 1) / 2) == len(ref_sent):
            for i in range(0, len(splits) - 1, 2):
                sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                      '[TEMP]' + splits[i + 1] + '[TEMP]', 1).replace('[ASP]', '')
                sample += ' !sent! ' + str(ref_sent[int(i / 2)])
                samples.append(sample.replace('[TEMP]', '[ASP]'))
        elif not ref_sent or int((len(splits) - 1) / 2) != len(ref_sent):
            if not ref_sent:
                print(_text, ' -> No the reference sentiment found')
            else:
                print(_text, ' -> Unequal length of reference sentiment and aspects, ignore the reference sentiment.')

            for i in range(0, len(splits) - 1, 2):
                sample = text.replace('[ASP]' + splits[i + 1] + '[ASP]',
                                      '[TEMP]' + splits[i + 1] + '[TEMP]', 1).replace('[ASP]', '')
                samples.append(sample.replace('[TEMP]', '[ASP]'))
        else:
            raise ValueError('Invalid Input:{}'.format(text))

        return samples

    def prepare_infer_sample(self, text: str):
        self.process_data(self.parse_sample(text))

    def prepare_infer_dataset(self, infer_file, ignore_error):
        lines = load_apc_datasets(infer_file)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        label_set = set()
        ex_id = 0
        if len(samples) > 1:
            it = tqdm(samples, postfix='building word indices...')
        else:
            it = samples
        for i, text in enumerate(it):
            try:
                # handle for empty lines in inferring_tutorials dataset_utils
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                # check for given polarity
                if '!sent!' in text:
                    text, polarity = text.split('!sent!')[0].strip(), text.split('!sent!')[1].strip()
                    polarity = polarity if polarity else LABEL_PADDING
                    text = text.replace('[PADDING]', '')

                else:
                    polarity = str(LABEL_PADDING)

                # simply add padding in case of some aspect is at the beginning or ending of a sentence
                text_left, aspect, text_right = text.split('[ASP]')
                text_left = text_left.replace('[PADDING] ', '')
                text_right = text_right.replace(' [PADDING]', '')
                text = text_left + ' ' + aspect + ' ' + text_right

                prepared_inputs = prepare_input_for_apc(self.opt, self.tokenizer, text_left, text_right, aspect, input_demands=self.opt.inputs_cols)

                text_raw = prepared_inputs['text_raw']
                aspect = prepared_inputs['aspect']
                aspect_position = prepared_inputs['aspect_position']
                text_bert_indices = prepared_inputs['text_bert_indices']
                text_raw_bert_indices = prepared_inputs['text_raw_bert_indices']
                aspect_bert_indices = prepared_inputs['aspect_bert_indices']
                lcfs_vec = prepared_inputs['lcfs_vec']
                lcf_vec = prepared_inputs['lcf_vec']

                validate_example(text_raw, aspect, polarity)

                if self.opt.model_name == 'dlcf_dca_bert' or self.opt.model_name == 'dlcfs_dca_bert':
                    configure_dlcf_spacy_model(self.opt)
                    prepared_inputs = prepare_input_for_dlcf_dca(self.opt, self.tokenizer, text_left, text_right, aspect)
                    dlcf_vec = prepared_inputs['dlcf_cdm_vec'] if self.opt.lcf == 'cdm' else prepared_inputs['dlcf_cdw_vec']
                    dlcfs_vec = prepared_inputs['dlcfs_cdm_vec'] if self.opt.lcf == 'cdm' else prepared_inputs['dlcfs_cdw_vec']
                    depend_vec = prepared_inputs['depend_vec']
                    depended_vec = prepared_inputs['depended_vec']
                data = {
                    'ex_id': ex_id,

                    'text_raw': text_raw,

                    'aspect': aspect,

                    'aspect_position': aspect_position,

                    'lca_ids': lcf_vec,  # the lca indices are the same as the refactored CDM (lcf != CDW or Fusion) lcf vec

                    'lcf_vec': lcf_vec if 'lcf_vec' in self.opt.inputs_cols else 0,

                    'lcfs_vec': lcfs_vec if 'lcfs_vec' in self.opt.inputs_cols else 0,

                    'dlcf_vec': dlcf_vec if 'dlcf_vec' in self.opt.inputs_cols else 0,

                    'dlcfs_vec': dlcfs_vec if 'dlcfs_vec' in self.opt.inputs_cols else 0,

                    'depend_vec': depend_vec if 'depend_vec' in self.opt.inputs_cols else 0,

                    'depended_vec': depended_vec if 'depended_vec' in self.opt.inputs_cols else 0,

                    'spc_mask_vec': build_spc_mask_vec(self.opt, text_raw_bert_indices)
                    if 'spc_mask_vec' in self.opt.inputs_cols else 0,

                    'text_bert_indices': text_bert_indices
                    if 'text_bert_indices' in self.opt.inputs_cols else 0,

                    'aspect_bert_indices': aspect_bert_indices
                    if 'aspect_bert_indices' in self.opt.inputs_cols else 0,

                    'text_raw_bert_indices': text_raw_bert_indices
                    if 'text_raw_bert_indices' in self.opt.inputs_cols else 0,

                    'polarity': polarity,
                }

                label_set.add(polarity)
                ex_id += 1
                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing: {} Error info:{}'.format(text, e))
                else:
                    raise RuntimeError('Catch Exception: {}, use ignore_error=True to remove error samples.'.format(e))

        self.opt.polarities_dim = len(label_set)

        if 'left_lcf_vec' in self.opt.inputs_cols or 'right_lcf_vec' in self.opt.inputs_cols \
            or 'left_lcfs_vec' in self.opt.inputs_cols or 'right_lcfs_vec' in self.opt.inputs_cols:
            all_data = build_sentiment_window(all_data, self.tokenizer, self.opt.similarity_threshold)
            for data in all_data:

                cluster_ids = []
                for pad_idx in range(self.opt.max_seq_len):
                    if pad_idx in data['cluster_ids']:
                        # print(data['polarity'])
                        cluster_ids.append(self.opt.label_to_index.get(self.opt.index_to_label.get(data['polarity'], 'N.A.'), -999))
                    else:
                        cluster_ids.append(-100)
                        # cluster_ids.append(3)

                data['cluster_ids'] = np.asarray(cluster_ids, dtype=np.int64)
                data['side_ex_ids'] = np.array(0)
                data['aspect_position'] = np.array(0)

        else:
            for data in all_data:
                data['aspect_position'] = np.array(0)

        self.all_data = all_data
        return all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)
