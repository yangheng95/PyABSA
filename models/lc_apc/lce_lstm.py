# -*- coding: utf-8 -*-
# file: lce_lstm.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class LCE_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LCE_LSTM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.embed_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.lc_embed = nn.Embedding(2, opt.embed_dim)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.classifier = nn.Linear(opt.embed_dim, 2)

    def forward(self, inputs):
        text_global_indices = inputs[0]
        text_local_indices = inputs[1]
        lce_ids = inputs[2]
        mask_matrix = inputs[3]

        x = self.embed(text_global_indices)
        y = self.embed(text_local_indices)

        if self.opt.lce and 'lca' in self.opt.model_name:
            lc_embedding = self.lc_embed(lce_ids)
            x = torch.mul(x, lc_embedding)

        # LCF layer
        y = torch.mul(y, mask_matrix)
        x = self.linear(torch.cat((x, y), -1))

        lce_logits = self.classifier(x)
        lce_logits = lce_logits.view(-1, 2)
        lce_ids = lce_ids.view(-1)

        x_len = torch.sum(text_global_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        if self.opt.lcp:
            return out, lce_logits, lce_ids
        else:
            return out
