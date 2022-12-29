# -*- coding: utf-8 -*-
# file: dataset_item.py
# time: 02/11/2022 18:56
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import os


# To replace the class defined in https://github.com/yangheng95/PyABSA/blob/release/pyabsa/functional/dataset/dataset_manager.py#L18,
# so that the inference script works on a custom dataset.
class DatasetItem(list):
    def __init__(self, dataset_name, dataset_items=None):
        """
        DatasetItem is used to construct a dataset for PyABSA. Each dataset has a name, you can merge multiple datasets into one dataset by "dataset_items"
        :param dataset_name: name of the dataset
        :param dataset_items: list of dataset names or file paths
        """
        super().__init__()
        if os.path.exists(dataset_name):
            # fprint('Construct DatasetItem from {}, assign dataset_name={}'.format(dataset_name, os.path.basename(dataset_name)))
            # Normalizing the dataset's name (or path) to not end with a '/' or '\'
            while dataset_name and dataset_name[-1] in ["/", "\\"]:
                dataset_name = dataset_name[:-1]

        # Naming the dataset with the normalized folder name only
        self.dataset_name = os.path.basename(dataset_name)

        # Creating the list of items if it does not exist
        if not dataset_items:
            dataset_items = dataset_name

        if not isinstance(dataset_items, list):
            self.append(dataset_items)
        else:
            for d in dataset_items:
                self.append(d)
        self.name = self.dataset_name
