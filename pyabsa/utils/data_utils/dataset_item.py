# -*- coding: utf-8 -*-
# file: dataset_item.py
# time: 02/11/2022 18:56
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
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
        Initializes a DatasetItem object with the given dataset_name and dataset_items.
        DatasetItem is used to construct a dataset for PyABSA. Each dataset has a name,
        you can merge multiple datasets into one dataset by "dataset_items". If dataset_name is a list,
        the dataset_name will be set to "Unnamed_Dataset" and the dataset_items will be set to dataset_name.

        :param dataset_name: The name of the dataset. Can be a string or a list of strings.
        :param dataset_items: The list of dataset names or file paths. Default is None.
        """
        self.name = None
        # If the dataset_name is a DatasetItem object, copy its attributes to this object
        if isinstance(dataset_name, DatasetItem):
            self.dataset_name = dataset_name.dataset_name
            self.name = dataset_name.name

            # Append all the items in dataset_items to this object
            for d in dataset_items:
                self.append(d)
        else:
            # Initialize a list object
            super().__init__()

            # If the dataset_name is a list, set dataset_items to dataset_name
            if isinstance(dataset_name, list):
                dataset_items = dataset_name
                dataset_name = "Unnamed_Dataset"

            # If the dataset_name is a valid file path, set dataset_name to the basename of the file path
            if os.path.exists(dataset_name):
                while dataset_name and dataset_name[-1] in ["/", "\\"]:
                    dataset_name = dataset_name[:-1]
                self.dataset_name = os.path.basename(dataset_name)
            else:
                # Set the dataset_name to the given value
                self.dataset_name = dataset_name

            # If dataset_items is None, set it to dataset_name
            if not dataset_items:
                dataset_items = dataset_name

            # Append the dataset_items to this object
            if not isinstance(dataset_items, list):
                self.append(dataset_items)
            else:
                for d in dataset_items:
                    self.append(d)

            # Set the name attribute to the dataset_name
            self.name = self.dataset_name

    def __str__(self):
        return self.name
