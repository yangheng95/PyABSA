# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import tqdm

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels


class GloVeTCDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )

        all_data = []

        label_set = set()

        for i in tqdm.tqdm(range(len(lines)), desc="preparing dataloader"):
            line = lines[i].strip().split("$LABEL$")
            text, label = line[0], line[1]
            text = text.strip().lower()
            label = label.strip().lower()
            text_indices = self.tokenizer.text_to_sequence(text)

            data = {
                "text_indices": text_indices,
                "label": label,
            }

            label_set.add(label)

            all_data.append(data)

        check_and_fix_labels(label_set, "label", all_data, self.config)
        self.config.output_dim = len(label_set)

        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train", **kwargs):
        super().__init__(config, tokenizer, dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
