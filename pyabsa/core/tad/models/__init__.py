# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 2021/8/8
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import pyabsa.core.tad.classic.__glove__.models
import pyabsa.core.tad.classic.__bert__.models


class GloVeTADModelList(list):
    TADLSTM = pyabsa.core.tad.classic.__glove__.models.TADLSTM

    def __init__(self):
        model_list = [self.TADLSTM]
        super().__init__(model_list)


class BERTTADModelList(list):
    TADBERT = pyabsa.core.tad.classic.__bert__.TADBERT

    def __init__(self):
        model_list = [self.TADBERT]
        super().__init__(model_list)
