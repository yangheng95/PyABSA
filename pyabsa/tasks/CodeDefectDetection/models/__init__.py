# -*- coding: utf-8 -*-
# file: __init__.py
# time: 02/11/2022 15:47
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


class BERTCDDModelList(list):
    from .__plm__.bert import BERT_MLP

    BERT_MLP = BERT_MLP
    BERT = BERT_MLP

    def __init__(self):
        super(BERTCDDModelList, self).__init__([self.BERT_MLP])


class GloVeCDDModelList(list):
    from .__classic__.lstm import LSTM

    LSTM = LSTM

    def __init__(self):
        super(GloVeCDDModelList, self).__init__([self.LSTM])
