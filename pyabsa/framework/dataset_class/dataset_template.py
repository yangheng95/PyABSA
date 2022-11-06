# -*- coding: utf-8 -*-
# file: dataset_template.py
# time: 02/11/2022 15:44
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch
from torch.utils.data import Dataset


class PyABSADataset(Dataset):
    data = []

    def __init__(self, config, tokenizer, dataset_type, **kwargs):
        super(PyABSADataset, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type

        if hasattr(config, 'dataset_dict'):
            self.load_data_from_dict(config.dataset_dict, **kwargs)
        elif hasattr(config, 'dataset_file'):
            self.load_data_from_file(config.dataset_file, **kwargs)
        else:
            raise ValueError('Please specify dataset_dict or dataset_file in config')

        self.covert_to_tensor(self.data)
        self.data = self.covert_to_tensor(self.data)

        self.config.pop('dataset_dict', None)

    @staticmethod
    def covert_to_tensor(data):
        for d in data:
            if isinstance(d, dict):
                for key, value in d.items():
                    try:
                        if len(value) > 1:
                            d[key] = torch.tensor(value)
                            # print(f'Convert {key} to tensor')
                    except Exception as e:
                        # print(f'Cannot convert {key} to tensor, {e}')
                        pass
            elif isinstance(d, list):
                for value in data:
                    PyABSADataset.covert_to_tensor(value)
        return data

    def load_data_from_dict(self, dataset_dict, **kwargs):
        raise NotImplementedError('Please implement load_data_from_dict() in your dataset class')

    def load_data_from_file(self, dataset_file, **kwargs):
        raise NotImplementedError('Please implement load_data_from_file() in your dataset class')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return f'PyABASDataset: {len(self.data)} samples'

    def __repr__(self):
        return self.__str__()