# -*- coding: utf-8 -*-
# file: bert.py
# time: 02/11/2022 15:48
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT_MLP(nn.Module):
    inputs = ['text_indices', 'rna_type']

    def __init__(self, bert, config):
        super(BERT_MLP, self).__init__()
        self.config = config
        self.bert = bert
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

        self.type_embed = nn.Embedding(4, self.config.embed_dim, padding_idx=0)
        self.linear = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)

        if self.config.sigmoid_regression:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        rna_type = inputs[1]

        rna_type_ids = self.bert(rna_type)['last_hidden_state']
        last_hidden_state = self.bert(text_raw_indices)['last_hidden_state']
        last_hidden_state = self.linear(torch.cat([last_hidden_state, rna_type_ids], dim=-1))

        # last_hidden_state = self.bert(text_raw_indices)['last_hidden_state']

        pooled_out = self.pooler(last_hidden_state)
        out = self.dense(pooled_out)
        if self.sigmoid:
            out = self.sigmoid(out)
        return out
