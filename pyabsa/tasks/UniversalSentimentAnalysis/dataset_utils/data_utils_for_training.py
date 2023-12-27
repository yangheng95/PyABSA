# -*- coding: utf-8 -*-
# file: data_utils.py
# time: 15/03/2023
# author: HENG YANG <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import json
import random

import pandas as pd
import tqdm

from .instruction import USAInstruction


class USATrainingDataset:
    def load_data_from_dict(self, dataset_dict, **kwargs):
        pass

    def load_data_from_file(self, dataset_file, **kwargs):
        from datasets import Dataset, DatasetDict

        instances = read_json(dataset_file[self.dataset_type])
        usa_instructor = USAInstruction()
        self.config.usa_instructor = usa_instructor
        all_data = []

        for i in tqdm.tqdm(range(len(instances)), desc="preparing dataloader"):
            instructed_input, labels = usa_instructor.encode_input(
                instances[i],
                examples=[random.choice(instances), random.choice(instances)],
            )
            all_data.append({"text": instructed_input, "labels": labels})

        # with open("usa_dataset.json", "w") as f:
        #     new_all_data = []
        #     for i in range(len(instances)):
        #         new_all_data.append(
        #             {
        #                 "instruction": all_data[i]["text"],
        #                 "input": "",
        #                 "output": "{"
        #                 + '"text": "{}", "labels": "{}"'.format(
        #                     instances[i]["text"], instances[i]["labels"]
        #                 )
        #                 + "}",
        #             }
        #         )
        #     json.dump(new_all_data, f, indent=4, sort_keys=True)

        huggingface_dataset = DatasetDict(
            {self.dataset_type: Dataset.from_pandas(pd.DataFrame(all_data))}
        )
        huggingface_dataset = huggingface_dataset.map(
            self.tokenize_function_inputs, batched=True
        )
        return huggingface_dataset

    def __init__(self, config, tokenizer, dataset_type="train", **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.tokenized_dataset = self.load_data_from_file(config.dataset_file, **kwargs)

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
