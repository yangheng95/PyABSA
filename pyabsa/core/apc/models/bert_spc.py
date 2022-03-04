# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class BERT_SPC(nn.Module):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.opt = opt
        self.encoder = Encoder(bert.config, opt)
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs['text_bert_indices']
        output = self.bert(text_bert_indices)
        pooled_output = output.last_hidden_state
        pooled_output = self.pooler(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return {'logits': logits, 'hidden_state': pooled_output}
