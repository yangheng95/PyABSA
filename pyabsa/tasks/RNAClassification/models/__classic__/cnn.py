# -*- coding: utf-8 -*-
# file: cnn.py
# time: 02/11/2022 15:48
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import torch
import torch.nn as nn

from torch.nn import Conv1d, MaxPool1d, Linear, Dropout, functional as F


class CNN(nn.Module):
    inputs = ['text_indices']

    def __init__(self, embedding_matrix, config):
        super(CNN, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.cnn = Conv1d(self.config.embed_dim, self.config.hidden_dim, kernel_size=self.config.kernel_size, padding=self.config.padding)
        self.pooling = MaxPool1d(self.config.max_seq_len - self.config.kernel_size + 1)
        self.dense = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        hidden_states = self.cnn(x.transpose(1, 2))
        pooled_states = self.pooling(hidden_states)
        transposed_states = pooled_states.transpose(1, 2)
        out = self.dense(transposed_states).sum(dim=1, keepdim=False)
        return out
