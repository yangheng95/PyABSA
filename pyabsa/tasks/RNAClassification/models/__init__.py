# -*- coding: utf-8 -*-
# file: __init__.py.py
# time: 02/11/2022 15:47
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

class GloVeRNACModelList(list):
    from .__classic__.cnn import CNN
    from .__classic__.lstm import LSTM
    from .__classic__.transformer import Transformer
    from .__classic__.mhsa import MHSA

    CNN = CNN
    LSTM = LSTM
    Transformer = Transformer
    MHSA = MHSA

    def __init__(self):
        super(GloVeRNACModelList, self).__init__(
            [
                self.CNN,
                self.LSTM,
                self.Transformer,
                self.MHSA
            ]
        )


class BERTRNACModelList(list):
    from .__plm__.bert import BERT_MLP

    BERT_MLP = BERT_MLP

    def __init__(self):
        super(BERTRNACModelList, self).__init__(
            [
                self.BERT_MLP
            ]
        )
