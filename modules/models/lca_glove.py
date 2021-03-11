# -*- coding: utf-8 -*-
# file: lca_glove.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn

import numpy as np
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention, BertConfig

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

class LCA_GLOVE(nn.Module):

    def __init__(self, embedding_matrix, opt):
        super(LCA_GLOVE, self).__init__()
        # Only few of the parameters are necessary in the config.json, such as hidden_size, num_attention_heads
        self.config = BertConfig.from_json_file("modules/utils/bert_config.json")
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lc_embed = nn.Embedding(2, opt.embed_dim)
        self.global_encoder1 = SelfAttention(self.config, opt)
        self.local_encoder1 = SelfAttention(self.config, opt)
        self.local_encoder2 = SelfAttention(self.config, opt)
        self.mha = SelfAttention(self.config, opt)
        self.pool = BertPooler(self.config)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.classifier = nn.Linear(opt.embed_dim, 2)

    def forward(self, inputs):
        text_global_indices = inputs[0]
        text_local_indices = inputs[1]
        lca_ids = inputs[2]
        mask_matrix = inputs[3]

        # embedding layer
        global_out = self.embed(text_global_indices)
        local_out = self.embed(text_local_indices)

        global_out = self.global_encoder1(global_out)

        # if self.opt.lca:
        #     lc_embedding = self.lc_embed(lca_ids)
        #     global_out = torch.mul(lc_embedding, global_out)
        lc_embedding = self.lc_embed(lca_ids)
        global_out = torch.mul(lc_embedding, global_out)
        local_out = self.local_encoder1(local_out)

        # dropout
        global_out = self.dropout(global_out).to(self.opt.device)
        local_out = self.dropout(local_out).to(self.opt.device)

        # LCF layer
        local_out = torch.mul(local_out, mask_matrix)
        local_out = self.local_encoder2(local_out)

        # dropout
        global_out = self.dropout(global_out).to(self.opt.device)
        local_out = self.dropout(local_out).to(self.opt.device)

        cat_features = torch.cat((local_out, global_out), dim=-1)
        cat_features = self.linear(cat_features)

        lca_logits = self.classifier(cat_features)
        lca_logits = lca_logits.view(-1, 2)
        lca_ids = lca_ids.view(-1)

        # output layer
        pooled_out = self.pool(cat_features)
        sen_logits = self.dense(pooled_out)

        if self.opt.lcp:
            return sen_logits, lca_logits, lca_ids
        else:
            return sen_logits
