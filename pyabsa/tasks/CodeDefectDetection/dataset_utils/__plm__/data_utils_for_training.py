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
                padding="do_not_pad",
                truncation=False,
            )
            code_ids = code_ids[1:-1]
            if self.dataset_type == "train" and label == "1":
                over_sample_num = self.config.get("over_sample_num", 2)
            else:
                over_sample_num = 1
            for _ in range(over_sample_num):
                # for x in range(len(code_ids) // ((self.config.max_seq_len - 2) // 2) + 1):
                #     _code_ids = code_ids[x * (self.config.max_seq_len - 2) // 2:
                #                          (x + 1) * (self.config.max_seq_len - 2) // 2 + (self.config.max_seq_len - 2) // 2]
                #     print(x * (self.config.max_seq_len - 2) // 2)
                #     print((x + 1) * (self.config.max_seq_len - 2) // 2 + (self.config.max_seq_len - 2) // 2)
                for x in range(len(code_ids) // (self.config.max_seq_len - 2) + 1):
                    _code_ids = code_ids[
                        x
                        * (self.config.max_seq_len - 2) : (x + 1)
                        * (self.config.max_seq_len - 2)
                    ]
                    _code_ids = pad_and_truncate(
                        _code_ids,
                        self.config.max_seq_len - 2,
                        value=self.tokenizer.pad_token_id,
                    )
                    if _code_ids:
                        all_data.append(
                            {
                                "ex_id": ex_id,
                                # "code": code_src,
                                "source_ids": [self.tokenizer.cls_token_id]
                                + _code_ids
                                + [self.tokenizer.eos_token_id],
                                "label": label,
                                "corrupt_label": 0,
                            }
                        )
                        label_set.add(label)
                        c_label_set.add(0)

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
                    if label == "0":
                        continue
                    if "$FEATURE$" in code_src:
                        code_src, feature = code_src.split("$FEATURE$")
                    code_src = _prepare_corrupt_code(code_src)
                    corrupt_code_ids = self.tokenizer.text_to_sequence(
                        code_src,
                        max_length=self.config.max_seq_len,
                        padding="do_not_pad",
                        truncation=False,
                    )
                    corrupt_code_ids = corrupt_code_ids[1:-1]
                    for x in range(
                        len(corrupt_code_ids) // (self.config.max_seq_len - 2) + 1
                    ):
                        _corrupt_code_ids = corrupt_code_ids[
                            x
                            * (self.config.max_seq_len - 2) : (x + 1)
                            * (self.config.max_seq_len - 2)
                        ]
                        _corrupt_code_ids = pad_and_truncate(
                            _corrupt_code_ids,
                            self.config.max_seq_len - 2,
                            value=self.tokenizer.pad_token_id,
                        )
                        if _corrupt_code_ids:
                            all_data.append(
                                {
                                    "ex_id": ex_id,
                                    # "code": code_src,
                                    "source_ids": [self.tokenizer.cls_token_id]
                                    + _corrupt_code_ids
                                    + [self.tokenizer.eos_token_id],
                                    "label": "-100",
                                    "corrupt_label": 1,
                                }
                            )
                            label_set.add("-100")
                            c_label_set.add(1)

        check_and_fix_labels(label_set, "label", all_data, self.config)
        self.config.output_dim = len(label_set)

        self.data = all_data

    def __init__(self, config, tokenizer, dataset_type="train", **kwargs):
        super().__init__(config, tokenizer, dataset_type, **kwargs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
