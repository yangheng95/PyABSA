# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from pyabsa.networks.dynamic_rnn import DynamicLSTM


class TADLSTM(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, embedding_matrix, config):
        super(TADLSTM, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.lstm = DynamicLSTM(
            self.config.embed_dim,
            self.config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dense1 = nn.Linear(self.config.hidden_dim, self.config.class_dim)
        self.dense2 = nn.Linear(self.config.hidden_dim, self.config.adv_det_dim)
        self.dense2 = nn.Linear(self.config.hidden_dim, self.config.class_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        sent_logits = self.dense1(h_n[0])
        advdet_logits = self.dense2(h_n[0])
        adv_tr_logits = self.dense2(h_n[0])
        return sent_logits, advdet_logits, adv_tr_logits
