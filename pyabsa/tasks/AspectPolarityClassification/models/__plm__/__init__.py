# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:48
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


class PLMAPCModelList(list):
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

    AOA_BERT = AOA_BERT
    ASGCN_BERT = ASGCN_BERT
    ATAE_LSTM_BERT = ATAE_LSTM_BERT
    Cabasc_BERT = Cabasc_BERT
    IAN_BERT = IAN_BERT
    LSTM_BERT = LSTM_BERT
    MemNet_BERT = MemNet_BERT
    MGAN_BERT = MGAN_BERT
    RAM_BERT = RAM_BERT
    TC_LSTM_BERT = TC_LSTM_BERT
    TD_LSTM_BERT = TD_LSTM_BERT
    TNet_LF_BERT = TNet_LF_BERT

    def __init__(self):
        super(PLMAPCModelList, self).__init__(
            [
                self.AOA_BERT,
                self.ASGCN_BERT,
                self.ATAE_LSTM_BERT,
                self.Cabasc_BERT,
                self.IAN_BERT,
                self.LSTM_BERT,
                self.MemNet_BERT,
                self.MGAN_BERT,
                self.RAM_BERT,
                self.TC_LSTM_BERT,
                self.TD_LSTM_BERT,
                self.TNet_LF_BERT,
            ]
        )

    def __str__(self):
        return str([model.__name__ for model in self])


class BERTBaselineAPCModelList(PLMAPCModelList):
    pass
