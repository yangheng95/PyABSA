# -*- coding: utf-8 -*-
# file: data_utils_for_training.py
# time: 13:30 2023/3/2
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.


import numpy as np
from pyabsa.utils.pyabsa_utils import check_and_fix_labels

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset


class ASTEDataset(PyABSADataset):
    def __init__(self, data_path, max_seq_len=128):
        self.data = self.load_data_from_file(data_path)
        self.max_seq_len = max_seq_len

    def load_data_from_dict(self, data_dict, **kwargs):
        pass

    def load_data_from_file(self, file_path, **kwargs):
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text, labels = line.split("\t")
                labels = eval(labels)
                lines.append((text, labels))

        all_data = []
        # record sentiment polarity type to update output_dim
        label_set = set()

        ex_id = 0

        for text, labels in lines:
            prepared_inputs = prepare_input_for_apc(
                text=text, aspect="", input_demands=["text_indices"], device="cpu"
            )
            text_indices = prepared_inputs["text_indices"]

            label_ids = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.int64)
            sentiment_mask = np.zeros(
                (self.max_seq_len, self.max_seq_len), dtype=np.int64
            )

            for label in labels:
                aspect_start, aspect_end = label[0]
                sentiment_start, sentiment_end = label[1]
                polarity = label[2]

                label_ids[aspect_start][sentiment_start] = 1
                label_ids[aspect_end][sentiment_end] = 2

                if polarity == "POS":
                    sentiment_mask[sentiment_start][sentiment_end] = 1
                elif polarity == "NEG":
                    sentiment_mask[sentiment_start][sentiment_end] = -1
                else:
                    sentiment_mask[sentiment_start][sentiment_end] = 0

            data = {
                "ex_id": ex_id,
                "text_indices": text_indices,
                "label_ids": label_ids,
                "sentiment_mask": sentiment_mask,
            }
            ex_id += 1

            label_set.update(set([label[2] for label in labels]))
            all_data.append(data)

        check_and_fix_labels(
            label_set, "sentiment polarity", all_data, config=self.config
        )
        self.config.output_dim = len(label_set)

        for data in all_data:
            data["label_ids"] = data["label_ids"].astype(np.int64)
            data["sentiment_mask"] = data["sentiment_mask"].astype(np.int64)

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx]["text_indices"],
            self.data[idx]["label_ids"],
            self.data[idx]["sentiment_mask"],
        )
