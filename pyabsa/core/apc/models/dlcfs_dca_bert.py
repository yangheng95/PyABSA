# -*- coding: utf-8 -*-
# file: apc_utils.py
# time: 2021/5/23 0023
# author: xumayi <xumayi@m.scnu.edu.cn>
# github: https://github.com/XuMayi
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


def weight_distrubute_local(bert_local_out, depend_weight, depended_weight, depend_vec, depended_vec, opt):
    bert_local_out2 = torch.zeros_like(bert_local_out)
    depend_vec2 = torch.mul(depend_vec, depend_weight.unsqueeze(2))
    depended_vec2 = torch.mul(depended_vec, depended_weight.unsqueeze(2))
    bert_local_out2 = bert_local_out2 + torch.mul(bert_local_out, depend_vec2) + torch.mul(bert_local_out, depended_vec2)
    for j in range(depend_weight.size()[0]):
        bert_local_out2[j][0] = bert_local_out[j][0]
    return bert_local_out2


class PointwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class DLCFS_DCA_BERT(nn.Module):
    inputs = ['text_bert_indices', 'text_raw_bert_indices', 'dlcfs_vec', 'depend_vec', 'depended_vec']

    def __init__(self, bert, opt):
        super(DLCFS_DCA_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global

        self.hidden = opt.embed_dim
        self.opt = opt
        self.opt.bert_dim = opt.embed_dim
        self.dropout = nn.Dropout(opt.dropout)
        self.bert_SA_ = Encoder(bert.config, opt)

        self.mean_pooling_double = PointwiseFeedForward(self.hidden * 2, self.hidden, self.hidden)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(self.hidden, opt.polarities_dim)

        self.dca_sa = nn.ModuleList()
        self.dca_pool = nn.ModuleList()
        self.dca_lin = nn.ModuleList()

        for i in range(opt.dca_layer):
            self.dca_sa.append(Encoder(bert.config, opt))
            self.dca_pool.append(BertPooler(bert.config))
            self.dca_lin.append(nn.Sequential(
                nn.Linear(opt.bert_dim, opt.bert_dim * 2),
                nn.GELU(),
                nn.Linear(opt.bert_dim * 2, 1),
                nn.Sigmoid())
            )

    def weight_calculate(self, sa, pool, lin, d_w, ded_w, depend_out, depended_out):
        depend_sa_out = sa(depend_out)
        depend_sa_out = self.dropout(depend_sa_out)
        depended_sa_out = sa(depended_out)
        depended_sa_out = self.dropout(depended_sa_out)

        depend_pool_out = pool(depend_sa_out)
        depend_pool_out = self.dropout(depend_pool_out)
        depended_pool_out = pool(depended_sa_out)
        depended_pool_out = self.dropout(depended_pool_out)

        depend_weight = lin(depend_pool_out)
        depend_weight = self.dropout(depend_weight)
        depended_weight = lin(depended_pool_out)
        depended_weight = self.dropout(depended_weight)

        for i in range(depend_weight.size()[0]):
            depend_weight[i] = depend_weight[i].item() * d_w[i].item()
            depended_weight[i] = depended_weight[i].item() * ded_w[i].item()
            weight_sum = depend_weight[i].item() + depended_weight[i].item()
            if weight_sum != 0:
                depend_weight[i] = (2 * depend_weight[i] / weight_sum) ** self.opt.dca_p
                if depend_weight[i] > 2:
                    depend_weight[i] = 2
                depended_weight[i] = (2 * depended_weight[i] / weight_sum) ** self.opt.dca_p
                if depended_weight[i] > 2:
                    depended_weight[i] = 2
            else:
                depend_weight[i] = 1
                depended_weight[i] = 1
        return depend_weight, depended_weight

    def forward(self, inputs):
        if self.opt.use_bert_spc:
            text_bert_indices = inputs['text_bert_indices']
        else:
            text_bert_indices = inputs['text_raw_bert_indices']
        text_local_indices = inputs['text_raw_bert_indices']
        lcf_matrix = inputs['dlcfs_vec'].unsqueeze(2)
        depend_vec = inputs['depend_vec'].unsqueeze(2)
        depended_vec = inputs['depended_vec'].unsqueeze(2)

        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']
        local_context_features = self.bert4local(text_local_indices)['last_hidden_state']

        bert_local_out = torch.mul(local_context_features, lcf_matrix)

        depend_weight = torch.ones(bert_local_out.size()[0])
        depended_weight = torch.ones(bert_local_out.size()[0])

        for i in range(self.opt.dca_layer):
            depend_out = torch.mul(bert_local_out, depend_vec)
            depended_out = torch.mul(bert_local_out, depended_vec)
            depend_weight, depended_weight = self.weight_calculate(self.dca_sa[i], self.dca_pool[i], self.dca_lin[i],
                                                                   depend_weight, depended_weight, depend_out,
                                                                   depended_out)
            bert_local_out = weight_distrubute_local(bert_local_out, depend_weight, depended_weight, depend_vec, depended_vec,
                                                     self.opt)

        out_cat = torch.cat((bert_local_out, global_context_features), dim=-1)
        out_cat = self.mean_pooling_double(out_cat)
        out_cat = self.bert_SA_(out_cat)
        out_cat = self.bert_pooler(out_cat)
        dense_out = self.dense(out_cat)
        return {'logits': dense_out, 'hidden_state': out_cat}
