# -*- coding: utf-8 -*-
# file: dataset_list.py
# time: 02/11/2022 19:43
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


class ProteinRDatasetList(list):
    """
    Protein Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(ProteinRDatasetList, self).__init__(self.__class__.__dict__.values())


class ProteinRegressionDatasetList(list):
    """
    Protein Sequence-based Regression Dataset Lists
    """

    def __init__(self):
        super(ProteinRegressionDatasetList, self).__init__(
            self.__class__.__dict__.values()
        )
