# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class BERT_SPC(nn.Module):
    inputs = [
        'text_bert_indices',
        'left_text_bert_indices',
        'right_text_bert_indices',
        'spc_mask_vec',
        'lcf_vec'
    ]

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert4central = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)

        self.post_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_window_3h = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.post_encoder = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):

        text_bert_indices = inputs['text_bert_indices']
        left_text_bert_indices = inputs['left_text_bert_indices']
        right_text_bert_indices = inputs['right_text_bert_indices']

        if not self.opt.lsa:
            hidden_state = self.bert4central(text_bert_indices)['last_hidden_state']
            dense_out = self.dense(self.bert_pooler(self.dropout(hidden_state)))
            return {'logits': dense_out, 'hidden_state': hidden_state}

        central_aspect_feature = self.bert4central(text_bert_indices)['last_hidden_state']
        central_aspect_feature = self.bert_pooler(central_aspect_feature)

        left_aspect_feature = self.bert4central(left_text_bert_indices)['last_hidden_state']
        left_aspect_feature = self.bert_pooler(left_aspect_feature)

        right_aspect_feature = self.bert4central(right_text_bert_indices)['last_hidden_state']
        right_aspect_feature = self.bert_pooler(right_aspect_feature)

        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            if self.opt.eta >= 0:
                cat_features = torch.cat(
                    (central_aspect_feature, self.opt.eta * left_aspect_feature, (1 - self.opt.eta) * right_aspect_feature), -1)
            else:
                cat_features = torch.cat((central_aspect_feature, left_aspect_feature, right_aspect_feature), -1)
            sent_out = self.linear_window_3h(cat_features)
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((central_aspect_feature, left_aspect_feature), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((central_aspect_feature, right_aspect_feature), -1))
        else:
            raise KeyError('Invalid parameter of aggregation window building:', self.opt.window)

        sent_out = self.dropout(sent_out)
        dense_out = self.dense(sent_out)

        return {'logits': dense_out, 'hidden_state': sent_out}
