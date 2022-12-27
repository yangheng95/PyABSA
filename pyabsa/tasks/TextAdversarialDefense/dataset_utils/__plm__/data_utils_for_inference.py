# -*- coding: utf-8 -*-
# file: data_utils_for_inference.py
# time: 02/11/2022 15:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import numpy as np
import tqdm
from findfile import find_dir, find_cwd_dir
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pyabsa.framework.dataset_class.dataset_template import PyABSADataset
from pyabsa.utils.file_utils.file_utils import load_dataset_from_file
from pyabsa.utils.pyabsa_utils import fprint


class BERTTADInferenceDataset(Dataset):

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
        all_data = []
        if len(samples) > 100:
            it = tqdm.tqdm(samples, description='preparing text classification inference dataloader...')
        else:
            it = samples
        for ex_id, text in enumerate(it):
            try:
                # handle for empty lines in inference datasets
                if text is None or '' == text.strip():
                    raise RuntimeError('Invalid Input!')

                if '$LABEL$' in text:
                    text, _, labels = text.strip().partition('$LABEL$')
                    text = text.strip()
                    if labels.count(',') == 2:
                        label, is_adv, adv_train_label = labels.strip().split(',')
                        label, is_adv, adv_train_label = label.strip(), is_adv.strip(), adv_train_label.strip()
                    elif labels.count(',') == 1:
                        label, is_adv = labels.strip().split(',')
                        label, is_adv = label.strip(), is_adv.strip()
                        adv_train_label = '-100'
                    elif labels.count(',') == 0:
                        label = labels.strip()
                        adv_train_label = '-100'
                        is_adv = '-100'
                    else:
                        label = '-100'
                        adv_train_label = '-100'
                        is_adv = '-100'

                    label = int(label)
                    adv_train_label = int(adv_train_label)
                    is_adv = int(is_adv)

                else:
                    text = text.strip()
                    label = -100
                    adv_train_label = -100
                    is_adv = -100

                text_indices = self.tokenizer.text_to_sequence('{}'.format(text))

                data = {
                    'text_indices': text_indices,

                    'text_raw': text,

                    'label': label,

                    'adv_train_label': adv_train_label,

                    'is_adv': is_adv,

                    # 'label': self.config.label_to_index.get(label, -100) if isinstance(label, str) else label,
                    #
                    # 'adv_train_label': self.config.adv_train_label_to_index.get(adv_train_label, -100) if isinstance(adv_train_label, str) else adv_train_label,
                    #
                    # 'is_adv': self.config.is_adv_to_index.get(is_adv, -100) if isinstance(is_adv, str) else is_adv,
                }

                all_data.append(data)

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
