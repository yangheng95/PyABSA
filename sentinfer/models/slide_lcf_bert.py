# -*- coding: utf-8 -*-
# file: slide_lcf_bert.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import copy

import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention


class Encoder(nn.Module):
    def __init__(self, config, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.config = config
        self.encoder = nn.ModuleList([SelfAttention(config, opt) for _ in range(1)])
        self.tanh = torch.nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        for i, enc in enumerate(self.encoder):
            x = self.tanh(enc(x)[0])
        return x


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return SA_out


class SLIDE_LCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SLIDE_LCF_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = copy.deepcopy(bert) if opt.use_dual_bert else self.bert4global
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.encoder = Encoder(bert.config, opt)
        self.encoder_left = Encoder(bert.config, opt)
        self.encoder_right = Encoder(bert.config, opt)
        self.linear2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_window_3h = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        spc_mask_vec = inputs[1]
        lcf_matrix = inputs[2]
        left_lcf_matrix = inputs[3]
        right_lcf_matrix = inputs[4]
        left_asp_dist_w = inputs[5]
        right_asp_dist_w = inputs[6]
        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']
        masked_global_context_features = torch.mul(spc_mask_vec, global_context_features)
        # # --------------------------------------------------- #
        lcf_features = torch.mul(masked_global_context_features, lcf_matrix)
        lcf_features = self.encoder(lcf_features)
        # # --------------------------------------------------- #
        left_lcf_features = torch.mul(masked_global_context_features, left_lcf_matrix)
        if 'slide' in self.opt.model_name:
            left_lcf_features = torch.mul(left_asp_dist_w, left_lcf_features)
        left_lcf_features = self.encoder_left(left_lcf_features)
        # # --------------------------------------------------- #
        right_lcf_features = torch.mul(masked_global_context_features, right_lcf_matrix)
        if 'slide' in self.opt.model_name:
            right_lcf_features = torch.mul(right_asp_dist_w, right_lcf_features)
        right_lcf_features = self.encoder_right(right_lcf_features)
        # # --------------------------------------------------- #

        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            sent_out = self.linear_window_3h(
                torch.cat((lcf_features, left_lcf_features, right_lcf_features), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, right_lcf_features), -1))
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, left_lcf_features), -1))
        sent_out = self.linear2(torch.cat((global_context_features, sent_out), -1))
        dense_out = self.dense(self.bert_pooler(sent_out))

        return dense_out
