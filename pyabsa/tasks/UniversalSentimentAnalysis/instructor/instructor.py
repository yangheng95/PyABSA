# -*- coding: utf-8 -*-
# file: apc_instructor.py
# time: 2021/4/22 0022
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import pickle

from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.tasks.UniversalSentimentAnalysis.dataset_utils.data_utils_for_training import (
    USATrainingDataset,
)
from pyabsa.utils.pyabsa_utils import fprint, print_args


class USATrainingInstructor(BaseTrainingInstructor):
    def _load_dataset_and_prepare_dataloader(self):
        cache_path = self.load_cache_dataset()

        # init BERT-based model and dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_bert)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.config.tokenizer = self.tokenizer

        if not os.path.exists(cache_path) or self.config.overwrite_cache:
            self.train_set = USATrainingDataset(
                self.config, self.tokenizer, dataset_type="train"
            ).tokenized_dataset
            self.test_set = USATrainingDataset(
                self.config, self.tokenizer, dataset_type="test"
            ).tokenized_dataset
            self.valid_set = USATrainingDataset(
                self.config, self.tokenizer, dataset_type="valid"
            ).tokenized_dataset

            self.save_cache_dataset(cache_path)
        else:
            fprint("Loading dataset from cache file: %s" % cache_path)
            with open(cache_path, "rb") as cache_path:
                (
                    self.train_set,
                    self.test_set,
                    self.valid_set,
                    self.config,
                ) = pickle.load(cache_path)
        # merge train datasets using datasets.DatasetDict
        self.datasets = {
            "train": self.train_set["train"],
            "test": self.test_set["test"],
            "valid": self.valid_set["valid"],
        }
        self.model = self.config.model(config=self.config)

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        print_args(self.config)

    def run(self):
        return self.model.train(self.datasets)
