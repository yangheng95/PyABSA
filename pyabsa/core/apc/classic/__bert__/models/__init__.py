# -*- coding: utf-8 -*-
# project: PyABSA
# file: __init__.py
# time: 2021/7/17
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from .aoa_bert import AOA_BERT
from .asgcn_bert import ASGCN_BERT
from .atae_lstm_bert import ATAE_LSTM_BERT
from .cabasc_bert import Cabasc_BERT
from .ian_bert import IAN_BERT
from .lstm_bert import LSTM_BERT
from .memnet_bert import MemNet_BERT
from .mgan_bert import MGAN_BERT
from .ram_bert import RAM_BERT
from .tc_lstm_bert import TC_LSTM_BERT
from .td_lstm_bert import TD_LSTM_BERT
from .tnet_lf_bert import TNet_LF_BERT

# print('The BERT-baseline model are derived base on replacing the GloVe embedding using BERT. '
#       'And the BERT-baseline model are under testing.')
