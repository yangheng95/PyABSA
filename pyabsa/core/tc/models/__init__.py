# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 2021/8/8
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import pyabsa.core.tc.classic.__glove__.models
import pyabsa.core.tc.classic.__bert__.models


class GloVeClassificationModelList:
    LSTM = pyabsa.core.tc.classic.__glove__.models.LSTM


class BERTClassificationModelList:
    BERT = pyabsa.core.tc.classic.__bert__.BERT
