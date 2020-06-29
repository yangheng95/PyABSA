# -*- coding: utf-8 -*-
# file: __init__.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

# LC-ABSA
# Most of the following models are easy to adapted to target-level tasks.
from models.lc_absa.bert_spc import BERT_SPC
from models.lc_absa.bert_base import BERT_BASE
from models.lc_absa.lcf_glove import LCF_GLOVE
from models.lc_absa.lcf_bert import LCF_BERT
from models.lc_absa.lce_lstm import LCE_LSTM
from models.lc_absa.lce_glove import LCE_GLOVE
from models.lc_absa.lce_bert import LCE_BERT
from models.lc_absa.hlcf_glove import HLCF_GLOVE
from models.lc_absa.hlcf_bert import HLCF_BERT
from models.lc_absa.lcf_atepc import LCF_ATEPC

# ABSA
from models.absa.lstm import LSTM
from models.absa.ian import IAN
from models.absa.memnet import MemNet
from models.absa.ram import RAM
from models.absa.td_lstm import TD_LSTM
from models.absa.tc_lstm import TC_LSTM
from models.absa.cabasc import Cabasc
from models.absa.atae_lstm import ATAE_LSTM
from models.absa.tnet_lf import TNet_LF
from models.absa.aoa import AOA
from models.absa.mgan import MGAN
