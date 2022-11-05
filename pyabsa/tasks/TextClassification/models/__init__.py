# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 02/11/2022 15:47
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

class BERTTCModelList(list):
    from .__plm__.bert import BERT_MLP

    BERT_MLP = BERT_MLP

    def __init__(self):
        super(BERTTCModelList, self).__init__(
            [
                self.BERT_MLP
             ]
        )


class GloVeTCModelList(list):
    from .__classic__.lstm import LSTM

    LSTM = LSTM

    def __init__(self):
        super(GloVeTCModelList, self).__init__(
            [
                self.LSTM
             ]
        )
