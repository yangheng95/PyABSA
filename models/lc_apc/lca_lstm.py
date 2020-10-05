# -*- coding: utf-8 -*-
# file: lca_lstm.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn

from layers.dynamic_rnn import DynamicLSTM


class LCA_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LCA_LSTM, self).__init__()
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
        lca_ids = inputs[2]
        mask_matrix = inputs[3]

        x = self.embed(text_global_indices)
        y = self.embed(text_local_indices)

        if self.opt.lca and 'lca' in self.opt.model_name:
            lc_embedding = self.lc_embed(lca_ids)
            x = torch.mul(x, lc_embedding)

        # LCF layer
        y = torch.mul(y, mask_matrix)
        x = self.linear(torch.cat((x, y), -1))

        lca_logits = self.classifier(x)
        lca_logits = lca_logits.view(-1, 2)
        lca_ids = lca_ids.view(-1)

        x_len = torch.sum(text_global_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        if self.opt.lcp:
            return out, lca_logits, lca_ids
        else:
            return out
