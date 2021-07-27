# -*- coding: utf-8 -*-
# @FileName: fast_lcf_bert_att.py
# @Time    : 2021/6/20 9:29
# @Author  : yangheng@m.scnu.edu.cn
# @github  : https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler
from pyabsa.network.sa_encoder import Encoder
import torch.nn.functional as F

class FAST_LCF_BERT_ATT(nn.Module):
    def __init__(self, bert, opt):
        super(FAST_LCF_BERT_ATT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = Encoder(bert.config, opt)
        self.linear3 = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.bert_SA_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        print('{} is a test model!'.format(self.__class__.__name__))

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs[0]
        else:
            text_bert_indices = inputs[1]
        text_local_indices = inputs[1]
        lcf_matrix = inputs[2]
        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']

        # LCF layer
        lcf_features = torch.mul(global_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        alpha_mat = torch.matmul(lcf_features, global_context_features.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        lcf_att_features = torch.matmul(alpha, global_context_features).squeeze(1)  # batch_size x 2*hidden_dim
        global_features = self.bert_pooler(global_context_features)
        lcf_features = self.bert_pooler(lcf_features)
        out = self.linear3(torch.cat((global_features, lcf_att_features, lcf_features), dim=-1))

        dense_out = self.dense(out)
        return dense_out
