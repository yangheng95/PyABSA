# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:35
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from pyabsa.utils.data_utils.dataset_item import DatasetItem


class ACOSDatasetList(list):
    """
    The following datasets are for aspect polarity classification task.
    The datasets are collected from different sources, you can use the id to locate the dataset.
    """

    Laptop14 = DatasetItem("Laptop14", "501.Laptop14")

    Restaurant14 = DatasetItem("Restaurant14", "502.Restaurant14")
    Restaurant15 = DatasetItem("Restaurant15", "503.Restaurant15")
    Restaurant16 = DatasetItem("Restaurant16", "504.Restaurant16")

    Chinese_Zhang = DatasetItem("Chinese_Zhang", "505.Chinese_Zhang")

    Synthetic = DatasetItem("Synthetic", "506.Synthetic")

    def __init__(self):
        super(ACOSDatasetList, self).__init__(
            [
                self.Laptop14,
                self.Restaurant14,
                self.Restaurant15,
                self.Restaurant16,
                self.Chinese_Zhang,
                self.Synthetic,
            ]
        )
