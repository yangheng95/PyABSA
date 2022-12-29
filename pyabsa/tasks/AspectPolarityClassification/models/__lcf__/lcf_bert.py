# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <hy345@exeter.ac.uk>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.networks.sa_encoder import Encoder


class LCF_BERT(nn.Module):
    inputs = ["text_indices", "text_raw_bert_indices", "lcf_vec"]

    def __init__(self, bert, config):
        super(LCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.bert_SA = Encoder(bert.config, config)
        self.linear2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.bert_SA_ = Encoder(bert.config, config)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(config.embed_dim, config.output_dim)

    def forward(self, inputs):
        if self.config.use_bert_spc:
            text_indices = inputs["text_indices"]
        else:
            text_indices = inputs["text_raw_bert_indices"]
        text_local_indices = inputs["text_raw_bert_indices"]
        lcf_matrix = inputs["lcf_vec"].unsqueeze(2)
        global_context_features = self.bert4global(text_indices)["last_hidden_state"]
        local_context_features = self.bert4local(text_local_indices)[
            "last_hidden_state"
        ]

        # LCF layer
        lcf_features = torch.mul(local_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        cat_features = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)
        cat_features = self.bert_SA_(cat_features)
        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)
        return {"logits": dense_out, "hidden_state": pooled_out}
