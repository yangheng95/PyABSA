# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import tqdm

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate


class BERTProteinRDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )

        all_data = []

        for ex_id, i in enumerate(
                tqdm.tqdm(range(len(lines)), desc="preparing dataloader")
        ):
            text, _, label = lines[i].partition("$LABEL$")
            seq, ph = text.split(",")
            label = float(label.strip())

            for x in range(len(seq) // (self.config.max_seq_len * 2) + 1):
                _seq = seq[
                       x
                       * (self.config.max_seq_len * 2): (x + 1)
                                                        * (self.config.max_seq_len * 2)
                       ]
                protein_indices = self.tokenizer.text_to_sequence(_seq)
                protein_indices = pad_and_truncate(
                    protein_indices,
                    self.config.max_seq_len,
                    value=self.tokenizer.pad_token_id,
                )
                data = {
                    "ex_id": torch.tensor(ex_id, dtype=torch.long),
                    "text_indices": torch.tensor(protein_indices, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.float32),
                }

                all_data.append(data)

        self.config.output_dim = 1

        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train"):
        super().__init__(config, tokenizer, dataset_type)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
