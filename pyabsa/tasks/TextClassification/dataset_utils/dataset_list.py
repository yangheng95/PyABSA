# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:41
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from pyabsa.utils.data_utils.dataset_item import DatasetItem


class TCDatasetList(list):
    """
    Text Classification or Sentiment analysis datasets
    """

    SST1 = DatasetItem("SST1", "200.SST1")
    SST5 = DatasetItem("SST5", "200.SST1")
    SST2 = DatasetItem("SST2", "201.SST2")
    AGNews10K = DatasetItem("AGNews10K", "204.AGNews10K")
    IMDB10K = DatasetItem("IMDB10K", "202.IMDB10K")
    SST = DatasetItem("SST", ["201.SST2"])

    def __init__(self):
        super(TCDatasetList, self).__init__(
            [self.SST1, self.SST2, self.SST5, self.AGNews10K, self.IMDB10K, self.SST]
        )


class TextClassificationDatasetList(TCDatasetList):
    pass
