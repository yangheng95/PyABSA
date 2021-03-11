# -*- coding: utf-8 -*-
# file: __init__.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

# Local context focus-based APC methods
from modules.models.lcf_bert import LCF_BERT
from modules.models.lcf_glove import LCF_GLOVE
from modules.models.lca_bert import LCA_BERT
from modules.models.lca_glove import LCA_GLOVE
from modules.models.lca_lstm import LCA_LSTM

# Communities-shared APC methods
from modules.models.aen import AEN_BERT
from modules.models.aoa import AOA
from modules.models.atae_lstm import ATAE_LSTM
from modules.models.cabasc import Cabasc
from modules.models.ian import IAN
from modules.models.lstm import LSTM
from modules.models.memnet import MemNet
from modules.models.mgan import MGAN
from modules.models.ram import RAM
from modules.models.tc_lstm import TC_LSTM
from modules.models.td_lstm import TD_LSTM
from modules.models.tnet_lf import TNet_LF
from modules.models.bert_base import BERT_BASE
from modules.models.bert_spc import BERT_SPC

