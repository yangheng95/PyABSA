# -*- coding: utf-8 -*-
# file: data_utils.py
# time: 15/03/2023
# author: HENG YANG <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import json
import os

import pandas as pd


class USAInferenceDataset:
    def prepare_infer_dataset(self, text, **kwargs):
        from datasets import Dataset, DatasetDict

        if isinstance(text, str) and os.path.exists(text):
            text = read_json(text)

        elif not isinstance(text, list):
            text = [text]

        for i, t in enumerate(text):
            try:
                text[i] = json.loads(t)
            except:
                pass

        all_data = []
        usa_instructor = self.config.usa_instructor
        for t in text:
            try:
                instructed_input, labels = usa_instructor.encode_input(
                    t,
                )
                all_data.append({"text": instructed_input, "labels": labels})
            except Exception as e:
                print(e)
                if kwargs.get("ignore_error", False):
                    continue
                else:
                    raise RuntimeError("Fail to encode the input text: {}".format(t))

        huggingface_dataset = DatasetDict(
            {self.dataset_type: Dataset.from_pandas(pd.DataFrame(all_data))}
        )
        huggingface_dataset = huggingface_dataset.map(
            self.tokenize_function_inputs, batched=True
        )
        self.tokenized_dataset = huggingface_dataset
        return huggingface_dataset

    def __init__(self, config, tokenizer, dataset_type="test", **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.tokenized_dataset = None

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(
            sample["text"], max_length=self.config.max_seq_len, truncation=True
        )
        labels = self.tokenizer(
            sample["labels"], max_length=self.config.max_seq_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def read_json(data_path):
    data = []

    for f in data_path:
        print(f)
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                data.append(json.loads(line))
    return data
