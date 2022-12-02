# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import tqdm
from torch.utils.data import Dataset

from pyabsa import LabelPaddingOption
from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.framework.tokenizer_class.tokenizer_class import pad_and_truncate
from pyabsa.utils.pyabsa_utils import fprint


class BERTRNACInferenceDataset(Dataset):

    def __init__(self, config, tokenizer, dataset_type='infer'):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.data = []

    def parse_sample(self, text):
        return [text]

    def prepare_infer_sample(self, text: str, ignore_error):
        self.process_data(self.parse_sample(text), ignore_error=ignore_error)

    def prepare_infer_dataset(self, infer_file, ignore_error):

        lines = load_dataset_from_file(infer_file, logger=self.config.logger)
        samples = []
        for sample in lines:
            if sample:
                samples.extend(self.parse_sample(sample))
        self.process_data(samples, ignore_error)

    def process_data(self, samples, ignore_error=True):
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, postfix='preparing text classification dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                text, _, label = text.strip().partition('$LABEL$')
                rna, rna_type = text.strip().split(',')
                label = label.strip() if label else LabelPaddingOption.LABEL_PADDING
                label = label.upper()
                rna_type_indices = self.tokenizer.text_to_sequence(str(rna_type))
                rna_indices = self.tokenizer.text_to_sequence(rna + ' ' + rna_type)

                data = {
                    'ex_id': ex_id,
                    'text_raw': rna,
                    'text_indices': rna_indices,
                    'rna_type': rna_type_indices,
                    'label': label,
                }
                all_data.append(data)

                self.data = all_data
            except Exception as e:
                if ignore_error:
                    fprint('Ignore error while processing:', text)
                else:
                    raise e

        self.data = all_data

        self.data = PyABSADataset.covert_to_tensor(self.data)

        return self.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GloVeRNACInferenceDataset(BERTRNACInferenceDataset):
    pass
