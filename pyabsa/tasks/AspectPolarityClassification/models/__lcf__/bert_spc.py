# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.sa_encoder import Encoder


class BERT_SPC(nn.Module):
    inputs = ['text_indices',
              'left_text_indices',
              'right_text_indices']

    def __init__(self, bert, config):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.config = config
        self.linear = nn.Linear(config.embed_dim, config.embed_dim)
        self.linear_window_2h = nn.Linear(2 * config.embed_dim, config.embed_dim) if self.config.lsa else nn.Linear(config.embed_dim, config.embed_dim)
        self.linear_window_3h = nn.Linear(3 * config.embed_dim, config.embed_dim) if self.config.lsa else nn.Linear(config.embed_dim, config.embed_dim)
        self.encoder = Encoder(bert.config, config)
        self.dropout = nn.Dropout(config.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.embed_dim, config.output_dim)

    def forward(self, inputs):
        res = {'logits': None}
        if self.config.lsa:
            feat = self.bert(inputs['text_indices'])['last_hidden_state']
            left_feat = self.bert(inputs['left_text_indices'])['last_hidden_state']
            right_feat = self.bert(inputs['right_text_indices'])['last_hidden_state']
            if 'lr' == self.config.window or 'rl' == self.config.window:
                if self.config.eta >= 0:
                    cat_features = torch.cat((feat, self.config.eta * left_feat, (1 - self.config.eta) * right_feat), -1)
                else:
                    cat_features = torch.cat((feat, left_feat, right_feat), -1)
                sent_out = self.linear_window_3h(cat_features)
            elif 'l' == self.config.window:
                sent_out = self.linear_window_2h(torch.cat((feat, left_feat), -1))
            elif 'r' == self.config.window:
                sent_out = self.linear_window_2h(torch.cat((feat, right_feat), -1))
            else:
                raise KeyError('Invalid parameter:', self.config.window)

            cat_feat = self.linear(sent_out)
            cat_feat = self.dropout(cat_feat)
            cat_feat = self.encoder(cat_feat)
            cat_feat = self.pooler(cat_feat)
            res['logits'] = self.dense(cat_feat)

        else:
            cat_feat = self.bert(inputs['text_indices'])['last_hidden_state']
            cat_feat = self.linear(cat_feat)
            cat_feat = self.dropout(cat_feat)
            cat_feat = self.encoder(cat_feat)
            cat_feat = self.pooler(cat_feat)
            res['logits'] = self.dense(cat_feat)

        return res
