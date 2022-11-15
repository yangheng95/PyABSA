# -*- coding: utf-8 -*-
# file: bert.py
# time: 02/11/2022 15:48
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT_MLP(nn.Module):
    inputs = ['text_indices']

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.config = config
        self.bert = bert
        self.pooler = BertPooler(bert.config)
        self.dense1 = nn.Linear(config.hidden_dim, config.output_dim1)
        self.dense2 = nn.Linear(config.hidden_dim, config.output_dim2)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.sigmoid_regression:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        last_hidden_state = self.bert(text_raw_indices)['last_hidden_state']
        pooled_out = self.pooler(last_hidden_state)
        out1, out2 = self.dense1(pooled_out), self.dense2(pooled_out)
        return out1, out2
