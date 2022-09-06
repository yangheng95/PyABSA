# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class TADBERT(nn.Module):
    inputs = ['text_bert_indices']

    def __init__(self, bert, opt):
        super(TADBERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.pooler = BertPooler(bert.config)
        self.dense1 = nn.Linear(self.opt.hidden_dim, self.opt.class_dim)
        self.dense2 = nn.Linear(self.opt.hidden_dim, self.opt.adv_det_dim)
        self.dense3 = nn.Linear(self.opt.hidden_dim, self.opt.class_dim)

        self.encoder1 = Encoder(self.bert.config, opt=opt)
        self.encoder2 = Encoder(self.bert.config, opt=opt)
        self.encoder3 = Encoder(self.bert.config, opt=opt)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        last_hidden_state = self.bert(text_raw_indices)['last_hidden_state']

        sent_logits = self.dense1(self.pooler(last_hidden_state))
        advdet_logits = self.dense2(self.pooler(last_hidden_state))
        adv_tr_logits = self.dense3(self.pooler(last_hidden_state))
        outputs = {
            'sent_logits': sent_logits,
            'advdet_logits': advdet_logits,
            'adv_tr_logits': adv_tr_logits,
            'last_hidden_state': last_hidden_state,
        }
        return outputs
