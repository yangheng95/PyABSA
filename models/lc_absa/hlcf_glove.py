# -*- coding: utf-8 -*-
# file: hlcf_glove.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn

import numpy as np
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


class HLCF_GLOVE(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(HLCF_GLOVE, self).__init__()
        # Only few of the parameters are necessary in the config.json, such as hidden_size, num_attention_heads, etc.
        self.config = BertConfig.from_json_file("config.json")
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lc_embed = nn.Embedding(2, opt.embed_dim)
        self.global_encoder1 = SelfAttention(self.config, opt)
        self.local_encoder1 = SelfAttention(self.config, opt)
        self.local_encoder2 = SelfAttention(self.config, opt)
        self.bert_SA = SelfAttention(self.config, opt)
        self.pool = BertPooler(self.config)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.classifier = nn.Linear(opt.embed_dim, 2)
        self.hlcf_linear = nn.Linear(opt.embed_dim * 3, opt.embed_dim)

    def forward(self, inputs):

        text_global_indices = inputs[0].to(self.opt.device)
        text_local_indices = inputs[1].to(self.opt.device)
        lcf_matrix3, lcf_matrix5, lcf_matrix10 = inputs[2]
        lcf_matrix3 = lcf_matrix3.to(self.opt.device)
        lcf_matrix5 = lcf_matrix5.to(self.opt.device)
        lcf_matrix10 = lcf_matrix10.to(self.opt.device)

        # embedding layer
        global_out = self.embed(text_global_indices)
        local_out = self.embed(text_local_indices)

        global_out = self.global_encoder1(global_out)
        local_out = self.local_encoder1(local_out)

        # dropout
        global_out = self.dropout(global_out).to(self.opt.device)
        local_out = self.dropout(local_out).to(self.opt.device)

        # H-LCF
        if 'cascade' in self.opt.hlcf:
            local_out = self.HLF_SA1(torch.mul(local_out, lcf_matrix10))
            local_out = self.HLF_SA2(torch.mul(local_out, lcf_matrix5))
            local_out = self.HLF_SA3(torch.mul(local_out, lcf_matrix3))
        elif 'parallel' in self.opt.hlcf:
            local_out3 = torch.mul(local_out, lcf_matrix3)
            local_out5 = torch.mul(local_out, lcf_matrix5)
            local_out10 = torch.mul(local_out, lcf_matrix10)
            local_out = self.hlcf_linear(torch.cat((local_out3, local_out5, local_out10), -1))

        # dropout
        global_out = self.dropout(global_out).to(self.opt.device)
        local_out = self.dropout(local_out).to(self.opt.device)

        cat_features = torch.cat((local_out, global_out), dim=-1)
        cat_features = self.linear(cat_features)
        # output layer
        pooled_out = self.pool(cat_features)
        dense_out = self.dense(pooled_out)

        return dense_out
