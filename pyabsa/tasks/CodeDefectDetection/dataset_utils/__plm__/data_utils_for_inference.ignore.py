# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import tqdm
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate
from torch.utils.data import Dataset

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import fprint
from ..cdd_utils import read_defect_examples


class BERTCDDInferenceDataset(Dataset):
    def __init__(self, config, tokenizer):

        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_dataset_from_file(infer_file, config=self.config)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        samples = read_defect_examples(
            samples,
            self.config.get("data_num", None),
            self.config.get("remove_comments", True),
        )
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, desc="preparing text classification dataloader")
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or "" == text.strip():
                    raise RuntimeError("Invalid Input!")

                code_src, label = text.strip().split("$LABEL$")
                if "$FEATURE$" in code_src:
                    code_src = code_src.split("$FEATURE$")[0]
                code_ids = self.tokenizer.tokenizer.tokenize(
                    code_src.replace("\n", ""),
                )

                # code_ids = self.tokenizer.text_to_sequence(
                #     code_src.replace("\n", ""),
                #     max_length=self.config.max_seq_len,
                #     padding="max_length",
                #     truncation=True,
                # )
                _code_ids = [self.tokenizer.cls_token] + code_ids[:self.config.max_seq_len - 2] + [
                    self.tokenizer.sep_token]
                _aux_ids = [self.tokenizer.cls_token] + code_ids[-self.config.max_seq_len + 2:] + [
                    self.tokenizer.sep_token]
                _code_ids = pad_and_truncate(_code_ids, self.config.max_seq_len, self.tokenizer.pad_token)
                _aux_ids = pad_and_truncate(_aux_ids, self.config.max_seq_len, self.tokenizer.pad_token)
                _code_ids = self.tokenizer.convert_tokens_to_ids(_code_ids)
                _aux_ids = self.tokenizer.convert_tokens_to_ids(_aux_ids)
                data = {
                    "ex_id": ex_id,
                    "code": code_src,
                    "source_ids": _code_ids,
                    "aux_ids": _aux_ids,
                    "label": int(label.strip()),
                    "corrupt_label": 0,
                }

                all_data.append(data)

            except Exception as e:
                if ignore_error:
                    fprint("Ignore error while processing:", text)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
