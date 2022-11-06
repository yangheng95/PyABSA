# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import tqdm
from torch.utils.data import Dataset

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate


class GloVeRNARDataset(Dataset):

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_dataset_from_file(infer_file, logger=self.config.logger)
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

                line = text.strip().split('\t') if '\t' in text else text.strip().split(',')
                try:
                    _, label, r1r2_label, r1r3_label, r2r3_label, seq = line[0], line[1], line[2], line[3], line[4], line[5]
                    label = float(label.strip())

                    # r1r2_label = float(r1r2_label.strip())
                    # r1r3_label = float(r1r3_label.strip())
                    # r2r3_label = float(r2r3_label.strip())
                    # if len(seq) > 2 * self.config.max_seq_len:
                    #     continue
                    for x in range(len(seq) // (self.config.max_seq_len * 3) + 1):
                        _seq = seq[x * (self.config.max_seq_len * 3):(x + 1) * (self.config.max_seq_len * 3)]
                        rna_indices = self.tokenizer.text_to_sequence(_seq)
                        rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)

                        data = {
                            'ex_id': torch.tensor(ex_id, dtype=torch.long),
                            'text_raw': seq,
                            'text_indices': torch.tensor(rna_indices, dtype=torch.long),
                            'label': torch.tensor(label, dtype=torch.float32),
                            # 'r1r2_label': torch.tensor(r1r2_label, dtype=torch.float32),
                            # 'r1r3_label': torch.tensor(r1r3_label, dtype=torch.float32),
                            # 'r2r3_label': torch.tensor(r2r3_label, dtype=torch.float32),
                        }

                        all_data.append(data)

                except Exception as e:
                    exon1, intron, exon2, label = line[0], line[1], line[2], line[3]
                    label = float(label.strip())
                    seq = exon1 + intron + exon2
                    exon1_ids = self.tokenizer.text_to_sequence(exon1)
                    intron_ids = self.tokenizer.text_to_sequence(intron)
                    exon2_ids = self.tokenizer.text_to_sequence(exon2)

                    intron_ids = pad_and_truncate(intron_ids, self.config.max_seq_len)

                    rna_indices = exon1_ids + intron_ids + exon2_ids
                    rna_indices = pad_and_truncate(rna_indices, self.config.max_seq_len)

                    data = {
                        'ex_id': torch.tensor(ex_id, dtype=torch.long),
                        'text_raw': seq,
                        'text_indices': torch.tensor(rna_indices, dtype=torch.long),
                        'intro_indices': torch.tensor(intron_ids, dtype=torch.long),
                        'label': torch.tensor(label, dtype=torch.float32)
                    }

                    all_data.append(data)

            except Exception as e:
                if ignore_error:
                    print('Ignore error while processing:', text, e)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
