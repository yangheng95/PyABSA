# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import random

import tqdm
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from ..cdd_utils import read_defect_examples, _prepare_corrupt_code
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import check_and_fix_labels, fprint


class BERTCDDDataset(PyABSADataset):
    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):
        lines = load_dataset_from_file(
            self.config.dataset_file[self.dataset_type], config=self.config
        )
        natural_examples = read_defect_examples(
            lines,
            self.config.get("data_num", None),
            self.config.get("remove_comments", True),
            tokenizer=self.tokenizer,
        )

        all_data = []

        label_set = set()
        c_label_set = set()

        for ex_id, line in enumerate(
            tqdm.tqdm(natural_examples, desc="preparing dataloader")
        ):
            code_src, label = line.strip().split("$LABEL$")
            if "$FEATURE$" in code_src:
                code_src, feature = code_src.split("$FEATURE$")
            # print(len(self.tokenizer.tokenize(code_src.replace('\n', ''))))
            code_ids = self.tokenizer.text_to_sequence(
                code_src,
                max_length=self.config.max_seq_len,
                padding="max_length",
                truncation=True,
            )
            aux_ids = self.tokenizer.text_to_sequence(
                feature,
                max_length=self.config.max_seq_len,
                padding="max_length",
                truncation=True,
            )
            data = {
                "ex_id": ex_id,
                "source_ids": code_ids,
                "aux_ids": aux_ids,
                "label": label,
                "corrupt_label": 0,
            }
            label_set.add(label)
            c_label_set.add(0)
            all_data.append(data)

        if self.dataset_type == "train":
            corrupt_examples = read_defect_examples(
                lines,
                self.config.get("data_num", None),
                self.config.get("remove_comments", True),
            )
            for _ in range(self.config.noise_instance_num):
                for ex_id, line in enumerate(
                    tqdm.tqdm(corrupt_examples, desc="preparing dataloader")
                ):
                    code_src, label = line.strip().split("$LABEL$")
                    if "$FEATURE$" in code_src:
                        code_src, feature = code_src.split("$FEATURE$")
                    code_src = _prepare_corrupt_code(code_src)
                    corrupt_code_ids = self.tokenizer.text_to_sequence(
                        code_src,
                        max_length=self.config.max_seq_len,
                        padding="max_length",
                        truncation=True,
                    )
                    aux_ids = self.tokenizer.text_to_sequence(
                        feature,
                        max_length=self.config.max_seq_len,
                        padding="max_length",
                        truncation=True,
                    )
                    data = {
                        "ex_id": ex_id,
                        "source_ids": corrupt_code_ids,
                        "aux_ids": aux_ids,
                        "label": "-100",
                        "corrupt_label": 1,
                    }
                    label_set.add("-100")
                    c_label_set.add(1)
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
