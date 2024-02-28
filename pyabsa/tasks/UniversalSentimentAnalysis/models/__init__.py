# -*- coding: utf-8 -*-
# file: __init__.py
# time: 13:52 06/12/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.


class USAModelList(list):
    from .model import GenerationModel

    GenerationModel = GenerationModel

    def __init__(self):
        super(USAModelList, self).__init__([self.GenerationModel])
