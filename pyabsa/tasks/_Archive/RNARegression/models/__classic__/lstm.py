# -*- coding: utf-8 -*-
# file: lstm.py
# time: 22/10/2022 17:33
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn

from pyabsa.networks.dynamic_rnn import DynamicLSTM


class LSTMLayer(nn.Module):
    def __init__(self, config):
        super(LSTMLayer, self).__init__()
        self.lstms = nn.ModuleList()
        self.config = config
        for i in range(self.config.num_lstm_layer):
            self.lstms.append(
                DynamicLSTM(
                    self.config.embed_dim,
                    self.config.hidden_dim,
                    num_layers=self.config.num_lstm_layer,
                    batch_first=True,
                    bidirectional=True,
                )
            )

    def forward(self, x, x_len):
        h, c = None, None
        for i in range(len(self.lstms)):
            x, (h, c) = self.lstms[i](x, x_len)
        return x, (h, c)


class LSTM(nn.Module):
    inputs = ["text_indices"]

    def __init__(self, embedding_matrix, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float)
        )
        self.lstm = LSTMLayer(config)
        self.dense = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out
