# -*- coding: utf-8 -*-
# file: __init__.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

# LC-APC
# Most of the following models are easy to adapted to target-level tasks.
from models.lc_apc.bert_spc import BERT_SPC
from models.lc_apc.bert_base import BERT_BASE
from models.lc_apc.lcf_glove import LCF_GLOVE
from models.lc_apc.lcf_bert import LCF_BERT
from models.lc_apc.lce_lstm import LCE_LSTM
from models.lc_apc.lce_glove import LCE_GLOVE
from models.lc_apc.lce_bert import LCE_BERT
from models.lc_apc.hlcf_glove import HLCF_GLOVE
from models.lc_apc.hlcf_bert import HLCF_BERT

# APC
from models.apc.lstm import LSTM
from models.apc.ian import IAN
from models.apc.memnet import MemNet
from models.apc.ram import RAM
from models.apc.td_lstm import TD_LSTM
from models.apc.tc_lstm import TC_LSTM
from models.apc.cabasc import Cabasc
from models.apc.atae_lstm import ATAE_LSTM
from models.apc.tnet_lf import TNet_LF
from models.apc.aoa import AOA
from models.apc.mgan import MGAN

# ATEPC
from models.lc_atepc.lcf_atepc import LCF_ATEPC