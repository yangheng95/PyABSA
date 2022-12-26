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

        lines = load_dataset_from_file(infer_file, config=self.config)
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
                label = label.strip().upper() if label else str(LabelPaddingOption.LABEL_PADDING)
                rna_type_indices = self.tokenizer.text_to_sequence(str(rna_type))
                rna_indices = self.tokenizer.text_to_sequence(rna + ' ' + rna_type, padding=False)

                for _ in range(self.config.get('noise_instances', 1)):
                    import numpy as np
                    _rna_indices = np.array(rna_indices.copy())

                    # noise_masks = np.abs(len(_rna_indices)//2-np.random.normal(loc=len(_rna_indices)//2, scale=self.config.max_seq_len//5, size=int(len(_rna_indices)*0.2)).astype(int))
                    # noise_masks = np.where(noise_masks < 0, 0, noise_masks)
                    # noise_masks = np.where(noise_masks > len(_rna_indices)-1, len(_rna_indices)-1, noise_masks)
                    # _rna_indices[noise_masks] = self.tokenizer.pad_token_id

                    # noise_masks = np.random.choice([0, 1], size=len(_rna_indices), p=[0.2, 0.8])
                    # _rna_indices = np.array(_rna_indices) * (
                    #     noise_masks if any(noise_masks) else [1] * len(_rna_indices))
                    # _rna_indices = _rna_indices.tolist()

                    _rna_indices = pad_and_truncate(_rna_indices, self.config.max_seq_len,
                                                    value=self.tokenizer.pad_token_id)

                    data = {
                        'ex_id': ex_id,
                        'text_raw': rna,
                        'text_indices': _rna_indices,
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
