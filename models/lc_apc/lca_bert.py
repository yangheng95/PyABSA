# -*- coding: utf-8 -*-
# file: lca_bert.py
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


class LCA_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(LCA_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert) if opt.use_dual_bert else self.bert4global
        self.lc_embed = nn.Embedding(opt.max_seq_len, opt.embed_dim)
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA_L = SelfAttention(bert.config, opt)
        self.bert_SA_G = SelfAttention(bert.config, opt)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.pool = BertPooler(bert.config)
        if self.opt.dataset in {'camera', 'notebook', 'car', 'phone'}:
            self.dense = nn.Linear(opt.embed_dim, 2)
        else:
            self.dense = nn.Linear(opt.embed_dim, 3)
        self.classifier = nn.Linear(opt.embed_dim, 2)

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_global_indices = inputs[0]
        else:
            text_global_indices = inputs[1]
        text_local_indices = inputs[1]
        bert_segments_ids = inputs[2]
        lca_ids = inputs[3]
        lcf_matrix = inputs[4]

        bert_global_out, _ = self.bert4global(text_global_indices, token_type_ids=bert_segments_ids)
        bert_local_out, _ = self.bert4local(text_local_indices)
        if self.opt.lca and 'lca' in self.opt.model_name:
            lc_embedding = self.lc_embed(lca_ids)
            bert_global_out = torch.mul(bert_global_out, lc_embedding)

        # # LCF-layer
        bert_local_out = torch.mul(bert_local_out, lcf_matrix)
        bert_local_out = self.bert_SA_L(bert_local_out)

        cat_features = torch.cat((bert_local_out, bert_global_out), dim=-1)
        cat_features = self.linear(cat_features)

        lca_logits = self.classifier(cat_features)
        lca_logits = lca_logits.view(-1, 2)
        lca_ids = lca_ids.view(-1)

        cat_features = self.dropout(cat_features)

        pooled_out = self.pool(cat_features)
        dense_out = self.dense(pooled_out)
        if self.opt.lcp:
            return dense_out, lca_logits, lca_ids
        else:
            return dense_out
