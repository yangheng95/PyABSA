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
    Restaurant16 = DatasetItem("Restaurant16", "502.Restaurant16")

    def __init__(self):
        super(ACOSDatasetList, self).__init__(
            [
                self.Laptop14,
                self.Restaurant16,
            ]
        )
