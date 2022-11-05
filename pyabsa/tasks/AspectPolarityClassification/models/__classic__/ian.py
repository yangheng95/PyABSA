# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from pyabsa.networks.attention import Attention
from pyabsa.networks.dynamic_rnn import DynamicLSTM


class IAN(nn.Module):
    inputs = ['text_indices', 'aspect_indices']

    def __init__(self, embedding_matrix, config):
        super(IAN, self).__init__()
        self.config = config
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(config.embed_dim, config.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(config.embed_dim, config.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(config.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(config.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(config.hidden_dim * 2, config.output_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs['text_indices'], inputs['aspect_indices']
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.embed(text_raw_indices)
        aspect = self.embed(aspect_indices)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.config.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = text_raw_len.clone().detach()
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return {'logits': out}
