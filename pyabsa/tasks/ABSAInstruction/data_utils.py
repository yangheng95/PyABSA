# -*- coding: utf-8 -*-
# file: data_utils.py
# time: 15/03/2023
# author: HENG YANG <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.

import json

import findfile
import pandas as pd

from .instruction import (
    ATEInstruction,
    CategoryInstruction,
    OpinionInstruction,
    APCInstruction,
)


class InstructDatasetLoader:
    def __init__(
            self,
            train_df_id,
            test_df_id,
            train_df_ood=None,
            test_df_ood=None,
            sample_size=1,
    ):
        self.train_df_id = train_df_id.sample(frac=sample_size, random_state=1999)
        self.test_df_id = test_df_id
        if train_df_ood is not None:
            self.train_df_ood = train_df_ood.sample(frac=sample_size, random_state=1999)
        else:
            self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood

    def prepare_instruction_dataloader(self, df):
        """
        Prepare the data in the input format required.
        """
        ate_instructor = ATEInstruction()
        apc_instructor = APCInstruction()
        op_instructor = OpinionInstruction()
        cat_instructor = CategoryInstruction()
        alldata = []
        for i, data in df.iterrows():
            _aspects = [label["aspect"] for label in data["labels"]]
            aspects = []
            for asp in _aspects:
                if asp.strip() not in aspects:
                    aspects.append(asp.strip())
            aspects = "|".join(aspects)

            polarities = []
            _polarities = [
                "{}:{}".format(label["aspect"], label["polarity"])
                for label in data["labels"]
            ]
            for pol in _polarities:
                if pol not in polarities:
                    polarities.append(pol)
            polarities = "|".join(polarities)

            opinions = "|".join(
                [
                    "{}:{}".format(label["aspect"], label["opinion"])
                    for label in data["labels"]
                ]
            )

            categories = "|".join(
                [
                    "{}:{}".format(label["aspect"], label["category"])
                    for label in data["labels"]
                ]
            )

            # ATE task
            alldata.append(
                {"text": ate_instructor.prepare_input(data["text"]), "labels": aspects}
            )

            # APC task
            alldata.append(
                {
                    "text": apc_instructor.prepare_input(data["text"], aspects),
                    "labels": polarities,
                }
            )

            # Opinion task
            alldata.append(
                {
                    "text": op_instructor.prepare_input(data["text"], aspects),
                    "labels": opinions,
                }
            )

            # Category task
            if "NULL" not in categories:
                alldata.append(
                    {
                        "text": cat_instructor.prepare_input(data["text"], aspects),
                        "labels": categories,
                    }
                )

        alldata = pd.DataFrame(alldata)
        return alldata

    def create_datasets(self, tokenize_function):
        from datasets import DatasetDict, Dataset

        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        if self.test_df_id is None:
            indomain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
        else:
            indomain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_id),
                    "test": Dataset.from_pandas(self.test_df_id),
                }
            )
        indomain_tokenized_datasets = indomain_dataset.map(
            tokenize_function, batched=True
        )

        if (self.train_df_ood is not None) and (self.test_df_ood is None):
            other_domain_dataset = DatasetDict(
                {"train": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {"test": Dataset.from_pandas(self.train_df_id)}
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        elif (self.train_df_ood is not None) and (self.test_df_ood is not None):
            other_domain_dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(self.train_df_ood),
                    "test": Dataset.from_pandas(self.test_df_ood),
                }
            )
            other_domain_tokenized_dataset = other_domain_dataset.map(
                tokenize_function, batched=True
            )
        else:
            other_domain_dataset = None
            other_domain_tokenized_dataset = None

        return (
            indomain_dataset,
            indomain_tokenized_datasets,
            other_domain_dataset,
            other_domain_tokenized_dataset,
        )


def read_json(data_path, data_type="train"):
    data = []

    files = findfile.find_files(data_path, [data_type, ".jsonl"], exclude_key=[".txt"])
    for f in files:
        print(f)
        with open(f, "r", encoding="utf8") as fin:
            for line in fin:
                data.append(json.loads(line))
    return data
