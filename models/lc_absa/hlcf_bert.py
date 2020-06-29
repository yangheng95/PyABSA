# -*- coding: utf-8 -*-
# file: hlcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn
import numpy as np
import copy

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


class HLCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(HLCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert) if opt.use_dual_bert else self.bert4global
        self.lc_embed = nn.Embedding(opt.max_seq_len, opt.embed_dim)
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA = SelfAttention(bert.config, opt)
        self.HLF_SA1 = SelfAttention(bert.config, opt)
        self.HLF_SA2 = SelfAttention(bert.config, opt)
        self.HLF_SA3 = SelfAttention(bert.config, opt)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.pool = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.classifier = nn.Linear(opt.embed_dim, 2)
        self.hlcf_linear = nn.Linear(opt.embed_dim * 3, opt.embed_dim)

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_global_indices = inputs[0].to(self.opt.device)
        else:
            text_global_indices = inputs[1].to(self.opt.device)
        text_local_indices = inputs[1].to(self.opt.device)
        bert_segments_ids = inputs[2].to(self.opt.device)
        lcf_matrix3, lcf_matrix5, lcf_matrix10 = inputs[3]
        lcf_matrix3 = lcf_matrix3.to(self.opt.device)
        lcf_matrix5 = lcf_matrix5.to(self.opt.device)
        lcf_matrix10 = lcf_matrix10.to(self.opt.device)
        bert_global_out, _ = self.bert4global(text_global_indices, token_type_ids=bert_segments_ids)
        bert_local_out, _ = self.bert4local(text_local_indices)

        # H-LCF
        if 'cascade' in self.opt.hlcf:
            bert_local_out = self.HLF_SA1(torch.mul(bert_local_out, lcf_matrix10))
            bert_local_out = self.HLF_SA2(torch.mul(bert_local_out, lcf_matrix5))
            bert_local_out = self.HLF_SA3(torch.mul(bert_local_out, lcf_matrix3))
        elif 'parallel' in self.opt.hlcf:
            bert_local_out3 = torch.mul(bert_local_out, lcf_matrix3)
            bert_local_out5 = torch.mul(bert_local_out, lcf_matrix5)
            bert_local_out10 = torch.mul(bert_local_out, lcf_matrix10)
            bert_local_out = self.hlcf_linear(torch.cat((bert_local_out3, bert_local_out5, bert_local_out10), -1))

        cat_features = torch.cat((bert_local_out, bert_global_out), dim=-1)
        cat_features = self.linear(cat_features)
        cat_features = self.dropout(cat_features)

        pooled_out = self.pool(cat_features)
        dense_out = self.dense(pooled_out)
        return dense_out
