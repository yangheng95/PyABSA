# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import tqdm

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate


class GloVeRNACDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        label_set = set()
        all_data = []

        for ex_id, data in enumerate(tqdm.tqdm(dataset_dict[self.dataset_type], postfix='preparing dataloader...')):
            if self.config.dataset_name.lower() in 'degrad':
                rna, label = data['text'], data['label']
                rna_indices = self.tokenizer.text_to_sequence(rna)
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)
                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)
            elif self.config.dataset_name.lower() in 'sfe':
                exon1, intron, exon2, label = data['exon1'], data['intron'], data['exon2'], data['label']
                exon1_ids = self.tokenizer.text_to_sequence(exon1, padding='do_not_pad')
                intron_ids = self.tokenizer.text_to_sequence(intron, padding='do_not_pad')
                exon2_ids = self.tokenizer.text_to_sequence(exon2, padding='do_not_pad')

                rna_indices = exon1_ids + intron_ids + exon2_ids
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)
                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)
        self.data = all_data

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(dataset_file[self.dataset_type])

        all_data = []

        label_set = set()

        for ex_id, i in enumerate(tqdm.tqdm(range(len(lines)), postfix='preparing dataloader...')):
            if self.config.dataset_name.lower() in 'degrad':
                text, _, label = lines[i].strip().partition('$LABEL$')
                rna = text.strip()
                label = label.strip()

                rna_indices = self.tokenizer.text_to_sequence(rna)

                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)

                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)

            elif self.config.dataset_name.lower() in 'sfe':
                text, _, label = lines[i].strip().partition('$LABEL$')
                label = label.strip() if label else LabelPaddingOption.LABEL_PADDING
                exon1, intron, exon2 = text.strip().split(',')
                exon1 = exon1.strip()
                intron = intron.strip()
                exon2 = exon2.strip()
                exon1_ids = self.tokenizer.text_to_sequence(exon1, padding='do_not_pad')
                intron_ids = self.tokenizer.text_to_sequence(intron, padding='do_not_pad')
                exon2_ids = self.tokenizer.text_to_sequence(exon2, padding='do_not_pad')
                rna_indices = [self.tokenizer.tokenizer.cls_token_id] + exon1_ids + intron_ids + exon2_ids + [self.tokenizer.tokenizer.sep_token_id]
                rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len, value=self.tokenizer.pad_token_id)

                intron_indices = self.tokenizer.text_to_sequence(intron)

                data = {
                    'ex_id': ex_id,
                    'text_indices': rna_indices,
                    'intron_indices': intron_indices,
                    'label': label,
                }
                label_set.add(label)
                all_data.append(data)

        check_and_fix_labels(label_set, 'label', all_data, self.config)
        self.config.output_dim = len(label_set)
        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type='train', **kwargs):
        super().__init__(config, tokenizer, dataset_type=dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
