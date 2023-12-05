# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import tqdm
from torch.utils.data import Dataset

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import fprint


class BERTRNARDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        if isinstance(text, list):
            self.process_data(text, ignore_error=ignore_error)
        else:
            self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):
        lines = load_dataset_from_file(infer_file, config=self.config)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
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

                seq, label = text.strip().split("$LABEL$")

                try:
                    label = float(label.strip())

                    # r1r2_label = float(r1r2_label.strip())
                    # r1r3_label = float(r1r3_label.strip())
                    # r2r3_label = float(r2r3_label.strip())
                    # if len(seq) > 2 * self.config.max_seq_len:
                    #     continue
                    for x in range(len(seq) // (self.config.max_seq_len * 2) + 1):
                        _seq = seq[
                               x
                               * (self.config.max_seq_len * 2): (x + 1)
                                                                * (self.config.max_seq_len * 2)
                               ]
                        # rna_indices = self.tokenizer.text_to_sequence(_seq)
                        rna_indices = self.tokenizer.convert_tokens_to_ids(list(seq))

                        data = {
                            "ex_id": torch.tensor(ex_id, dtype=torch.long),
                            "text_indices": torch.tensor(rna_indices, dtype=torch.long),
                            "text_raw": seq,
                            "label": torch.tensor(label, dtype=torch.float32),
                            # 'r1r2_label': torch.tensor(r1r2_label, dtype=torch.float32),
                            # 'r1r3_label': torch.tensor(r1r3_label, dtype=torch.float32),
                            # 'r2r3_label': torch.tensor(r2r3_label, dtype=torch.float32),
                        }
                        all_data.append(data)

                except Exception as e:
                    rna_seq, _, label = text.strip().partition("$LABEL$")

                    label = float(label.strip())
                    rna_indices = self.tokenizer.text_to_sequence(
                        rna_seq, padding="do_not_pad"
                    )
                    rna_indices = pad_and_truncate(
                        rna_indices,
                        self.config.max_seq_len,
                        value=self.tokenizer.pad_token_id,
                    )

                    data = {
                        "ex_id": torch.tensor(ex_id, dtype=torch.long),
                        "text_indices": torch.tensor(rna_indices, dtype=torch.long),
                        "text_raw": seq,
                        "label": torch.tensor(label, dtype=torch.float32),
                        # 'r1r2_label': torch.tensor(r1r2_label, dtype=torch.float32),
                        # 'r1r3_label': torch.tensor(r1r3_label, dtype=torch.float32),
                        # 'r2r3_label': torch.tensor(r2r3_label, dtype=torch.float32),
                    }

                    all_data.append(data)

            except Exception as e:
                if ignore_error:
                    fprint("Ignore error while processing:", text, e)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
