# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class BERT_SPC(nn.Module):
    inputs = ['text_bert_indices',
              'left_text_bert_indices',
              'right_text_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.opt = opt
        self.linear = nn.Linear(3 * opt.embed_dim, opt.embed_dim) if self.opt.lsa else nn.Linear(opt.embed_dim, opt.embed_dim)
        self.encoder = Encoder(bert.config, opt)
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        res = {'logits': None}
        if self.opt.lsa:
            c_feat = self.bert(inputs['text_bert_indices'])['last_hidden_state']
            l_feat = self.bert(inputs['left_text_bert_indices'])['last_hidden_state']
            r_feat = self.bert(inputs['right_text_bert_indices'])['last_hidden_state']
            cat_feat = torch.cat((c_feat, l_feat, r_feat), -1)
            cat_feat = self.linear(cat_feat)
            cat_feat = self.dropout(cat_feat)
            cat_feat = self.encoder(cat_feat)
            cat_feat = self.pooler(cat_feat)
            res['logits'] = self.dense(cat_feat)

        else:
            cat_feat = self.bert(inputs['text_bert_indices'])['last_hidden_state']
            cat_feat = self.linear(cat_feat)
            cat_feat = self.dropout(cat_feat)
            cat_feat = self.encoder(cat_feat)
            cat_feat = self.pooler(cat_feat)
            res['logits'] = self.dense(cat_feat)

        return res

