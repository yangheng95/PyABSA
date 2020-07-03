# -*- coding: utf-8 -*-
# file: hlcf_glove.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn

import numpy as np
from layers.point_wise_feed_forward import PositionwiseFeedForward
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig


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


class LCF_GLOVE(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(LCF_GLOVE, self).__init__()
        self.config = BertConfig.from_json_file("utils/bert_config.json")
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.mha_global = SelfAttention(self.config, opt)
        self.mha_local = SelfAttention(self.config, opt)
        self.ffn_global = PositionwiseFeedForward(self.opt.embed_dim, dropout=self.opt.dropout)
        self.ffn_local = PositionwiseFeedForward(self.opt.embed_dim, dropout=self.opt.dropout)
        self.mha_local_SA = SelfAttention(self.config, opt)
        self.mha_global_SA = SelfAttention(self.config, opt)
        self.pool = BertPooler(self.config)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_global_indices = inputs[0]
        text_local_indices = inputs[1]
        lcf_matrix = inputs[2]

        # embedding layer
        global_context_features = self.embed(text_global_indices)
        local_context_features = self.embed(text_local_indices)

        # PFE layer
        global_context_features = self.mha_global(global_context_features)
        local_context_features = self.mha_local(local_context_features)
        global_context_features = self.ffn_global(global_context_features)
        local_context_features = self.ffn_local(local_context_features)

        # dropout
        global_context_features = self.dropout(global_context_features).to(self.opt.device)
        local_context_features = self.dropout(local_context_features).to(self.opt.device)

        # LCF layer
        local_context_features = torch.mul(local_context_features, lcf_matrix)
        lcf_features = self.mha_local_SA(local_context_features)

        global_context_features = self.mha_global_SA(global_context_features)
        # FIL layer
        cat_out = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_out = self.linear(cat_out)

        # output layer
        pooled_out = self.pool(cat_out)
        dense_out = self.dense(pooled_out)
        return dense_out
