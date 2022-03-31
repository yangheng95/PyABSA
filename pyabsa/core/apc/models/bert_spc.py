# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class BERT_SPC_Unit(nn.Module):

    def __init__(self, bert, opt):
        super(BERT_SPC_Unit, self).__init__()
        self.bert = bert
        self.opt = opt
        self.encoder = Encoder(bert.config, opt)
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        # self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs
        output = self.bert(text_bert_indices)
        pooled_output = output.last_hidden_state
        # pooled_output = self.pooler(pooled_output)
        # pooled_output = self.dropout(pooled_output)
        # logits = self.dense(pooled_output)
        # return {'logits': logits, 'hidden_state': pooled_output}
        return pooled_output


class BERT_SPC(nn.Module):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()

        self.opt = opt
        self.bert_left = BERT_SPC_Unit(bert, opt) if self.opt.lsa else None
        self.bert_central = BERT_SPC_Unit(bert, opt)
        self.bert_right = BERT_SPC_Unit(bert, opt) if self.opt.lsa else None
        # self.dense = nn.Linear(3 * opt.embed_dim, opt.polarities_dim) if self.opt.lsa else nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.dense = nn.Linear(3 * opt.embed_dim, opt.embed_dim) if self.opt.lsa else nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        self.logit_out = nn.Linear(opt.embed_dim, opt.polarities_dim)


    def forward(self, inputs):
        res = {'logits': None}
        if self.opt.lsa:
            cat_feat = torch.cat(
                (self.bert_left(inputs['text_bert_indices']),
                 self.bert_central(inputs['text_bert_indices']),
                 self.bert_right(inputs['text_bert_indices'])),
                -1)
            res['logits'] = self.dense(cat_feat)
            res['logits'] = self.logit_out(self.pooler(self.dropout(res['logits'])))

        else:
            res['logits'] = self.dense(self.bert_central(inputs['text_bert_indices']))
            res['logits'] = self.logit_out(self.pooler(self.dropout(res['logits'])))

        return res

