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
        self.linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.linear_window_2h = nn.Linear(2 * opt.embed_dim, opt.embed_dim) if self.opt.lsa else nn.Linear(opt.embed_dim, opt.embed_dim)
        self.linear_window_3h = nn.Linear(3 * opt.embed_dim, opt.embed_dim) if self.opt.lsa else nn.Linear(opt.embed_dim, opt.embed_dim)
        self.encoder = Encoder(bert.config, opt)
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)

        self.eta1 = nn.Parameter(torch.tensor(self.opt.eta, dtype=torch.float))
        self.eta2 = nn.Parameter(torch.tensor(self.opt.eta, dtype=torch.float))

        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        res = {'logits': None}
        if self.opt.lsa:
            c_feat = self.bert(inputs['text_bert_indices'])['last_hidden_state']
            l_feat = self.bert(inputs['left_text_bert_indices'])['last_hidden_state']
            r_feat = self.bert(inputs['right_text_bert_indices'])['last_hidden_state']
            if 'lr' == self.opt.window or 'rl' == self.opt.window:
                if self.opt.eta >= 0:
                    cat_features = torch.cat((self.eta1 * l_feat, c_feat, self.opt.eta2 * r_feat), -1)
                else:
                    cat_features = torch.cat((l_feat, c_feat, r_feat), -1)
                cat_feat = self.linear_window_3h(cat_features)
            elif 'l' == self.opt.window:
                cat_feat = self.linear_window_2h(torch.cat((l_feat, c_feat), -1))
            elif 'r' == self.opt.window:
                cat_feat = self.linear_window_2h(torch.cat((c_feat, r_feat), -1))
            else:
                raise KeyError('Invalid parameter:', self.opt.window)

            # cat_feat = torch.cat((c_feat, l_feat, r_feat), -1)
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
