# -*- coding: utf-8 -*-
# file: lca_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import copy

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class LCA_BERT(nn.Module):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'lcf_vec', 'polarity']

    def __init__(self, bert, opt):
        super(LCA_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert)
        self.lc_embed = nn.Embedding(2, opt.embed_dim)
        self.lc_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA_L = Encoder(bert.config, opt)
        self.bert_SA_G = Encoder(bert.config, opt)
        self.cat_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.pool = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)
        self.classifier = nn.Linear(opt.embed_dim, 2)
        self.lca_criterion = nn.CrossEntropyLoss()
        self.classification_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_global_indices = inputs['text_bert_indices']
        else:
            text_global_indices = inputs['text_raw_bert_indices']
        text_local_indices = inputs['text_raw_bert_indices']
        lca_ids = inputs['lcf_vec'].long()
        lcf_matrix = lca_ids.unsqueeze(2)  # lca_ids is the same as lcf_matrix
        polarity = inputs['polarity'] if 'polarity' in inputs else None

        bert_global_out = self.bert4global(text_global_indices)['last_hidden_state']
        bert_local_out = self.bert4local(text_local_indices)['last_hidden_state']

        lc_embedding = self.lc_embed(lca_ids)
        bert_global_out = self.lc_linear(torch.cat((bert_global_out, lc_embedding), -1))

        # # LCF-layer
        bert_local_out = torch.mul(bert_local_out, lcf_matrix)
        bert_local_out = self.bert_SA_L(bert_local_out)

        cat_features = torch.cat((bert_local_out, bert_global_out), dim=-1)
        cat_features = self.cat_linear(cat_features)

        lca_logits = self.classifier(cat_features)
        lca_logits = lca_logits.view(-1, 2)
        lca_ids = lca_ids.view(-1)

        cat_features = self.dropout(cat_features)

        pooled_out = self.pool(cat_features)
        sent_logits = self.dense(pooled_out)

        if polarity is not None:
            lcp_loss = self.lca_criterion(lca_logits, lca_ids)
            sent_loss = self.classification_criterion(sent_logits, polarity)
            return {'logits': sent_logits, 'hidden_state': pooled_out, 'loss': (1 - self.opt.sigma) * sent_loss + self.opt.sigma * lcp_loss}
        else:
            return {'logits': sent_logits, 'hidden_state': pooled_out}
