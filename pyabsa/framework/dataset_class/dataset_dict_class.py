# -*- coding: utf-8 -*-
# file: dataset_dict_class.py
# time: 05/11/2022 14:17
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

class DatasetDict(dict):
    def __init__(self, dataset_dict, *args, **kwargs):
        """
        A dict-like object for storing datasets

        dataset_dict = {
            'train': [
                {'data': 'This is a text for training', 'label': 'Positive'},
                {'data': 'This is a text for training', 'label': 'Negative'},
            ],
            'test': [
                {'data': 'This is a text for testing', 'label': 'Positive'},
                {'data': 'This is a text for testing', 'label': 'Negative'},
            ],
            'valid': [
                {'data': 'This is a text for validation', 'label': 'Positive'},
                {'data': 'This is a text for validation', 'label': 'Negative'},
            ],
            'dataset_name': str(),
            'label_name': str(),
        }
        """

        super().__init__(*args, **kwargs)
        self.update(dataset_dict)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

    def __getitem__(self, item):
        return super().__getitem__(item)
