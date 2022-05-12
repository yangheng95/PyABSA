# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 2021/8/8
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import pyabsa.core.ao_tc.classic.__glove__.models
import pyabsa.core.ao_tc.classic.__bert__.models


class AOGloVeTCModelList(list):
    LSTM = pyabsa.core.ao_tc.classic.__glove__.models.LSTM

    def __init__(self):
        model_list = [self.LSTM]
        super().__init__(model_list)


class AOBERTTCModelList(list):
    AOBERT = pyabsa.core.ao_tc.classic.__bert__.AOBERT

    def __init__(self):
        model_list = [self.AOBERT]
        super().__init__(model_list)
