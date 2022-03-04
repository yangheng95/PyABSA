# -*- coding: utf-8 -*-
# file: bert_base.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler


class BERT_BASE(nn.Module):
    inputs = ['text_raw_bert_indices']

    def __init__(self, bert, opt):
        super(BERT_BASE, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs['text_raw_bert_indices']
        text_features = self.bert(text_bert_indices)['last_hidden_state']
        pooled_output = self.pooler(text_features)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return {'logits': logits, 'hidden_state': pooled_output}
