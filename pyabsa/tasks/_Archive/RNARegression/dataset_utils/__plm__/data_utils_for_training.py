# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import json

import torch
import tqdm

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file


class BERTRNARDataset(PyABSADataset):
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
            line = json.loads(lines[i].strip())
            transcript = list(line["seq"].strip())
            structure = [int(_) for _ in list(line["stru2"].strip())]
            label = line["label"]

            assert len(transcript) == 100
            rna_indices = self.tokenizer.tokenizer.convert_tokens_to_ids(transcript)
            rna_indices = pad_and_truncate(
                rna_indices,
                self.config.max_seq_len,
                value=self.tokenizer.pad_token_id,
            )
            structure = pad_and_truncate(
                structure,
                self.config.max_seq_len,
                value=9,
            )
            data = {
                "ex_id": torch.tensor(ex_id, dtype=torch.long),
                "text_indices": torch.tensor(rna_indices, dtype=torch.long),
                "structure": torch.tensor(structure, dtype=torch.float32),
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
