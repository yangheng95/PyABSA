# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(LCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert) if opt.use_dual_bert else self.bert4global
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.linear2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear3 = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs[0]
        else:
            text_bert_indices = inputs[1]
        text_local_indices = inputs[1]
        bert_segments_ids = inputs[2]
        lcf_matrix = inputs[3]

        global_context_features, _ = self.bert4global(text_bert_indices, token_type_ids=bert_segments_ids)
        local_context_features, _ = self.bert4local(text_local_indices)

        # LCF layer
        lcf_features = torch.mul(local_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        cat_features = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)

        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)
        return dense_out
