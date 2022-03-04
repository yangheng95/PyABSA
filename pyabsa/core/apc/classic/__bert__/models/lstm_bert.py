# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from ..layers.dynamic_rnn import DynamicLSTM


class LSTM_BERT(nn.Module):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(LSTM_BERT, self).__init__()
        self.embed = bert
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs['text_bert_indices']
        x = self.embed(text_raw_indices)['last_hidden_state']
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return {'logits': out}
