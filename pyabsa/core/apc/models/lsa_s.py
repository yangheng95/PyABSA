# -*- coding: utf-8 -*-
# file: lsa_s.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.
import copy

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class LSA_S(nn.Module):
    inputs = ['text_bert_indices', 'left_text_bert_indices', 'right_text_bert_indices', 'spc_mask_vec', 'lcfs_vec', 'left_lcfs_vec', 'right_lcfs_vec']

    def __init__(self, bert, opt):
        super(LSA_S, self).__init__()
        self.bert4central = bert
        # self.bert4side = copy.deepcopy(bert)
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)

        self.encoder = Encoder(bert.config, opt)
        self.encoder_left = Encoder(bert.config, opt)
        self.encoder_right = Encoder(bert.config, opt)

        self.post_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_window_3h = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        self.post_encoder = Encoder(bert.config, opt)
        self.post_encoder_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs['text_bert_indices']
        left_text_bert_indices = inputs['left_text_bert_indices']
        right_text_bert_indices = inputs['right_text_bert_indices']
        spc_mask_vec = inputs['spc_mask_vec']
        lcf_matrix = inputs['lcfs_vec'].unsqueeze(2)
        left_lcf_matrix = inputs['left_lcfs_vec'].unsqueeze(2)
        right_lcf_matrix = inputs['right_lcfs_vec'].unsqueeze(2)

        global_context_features = self.bert4central(text_bert_indices)['last_hidden_state']
        left_global_context_features = self.bert4central(left_text_bert_indices)['last_hidden_state']
        right_global_context_features = self.bert4central(right_text_bert_indices)['last_hidden_state']
        # left_global_context_features = self.bert4side(left_text_bert_indices)['last_hidden_state']
        # right_global_context_features = self.bert4side(right_text_bert_indices)['last_hidden_state']

        # # --------------------------------------------------- #
        lcf_features = torch.mul(global_context_features, lcf_matrix)
        lcf_features = self.encoder(lcf_features)
        # # --------------------------------------------------- #
        left_lcf_features = torch.mul(left_global_context_features, left_lcf_matrix)
        left_lcf_features = self.encoder_left(left_lcf_features)
        # # --------------------------------------------------- #
        right_lcf_features = torch.mul(right_global_context_features, right_lcf_matrix)
        right_lcf_features = self.encoder_right(right_lcf_features)
        # # --------------------------------------------------- #

        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            if self.opt.eta >= 0:
                cat_features = torch.cat(
                    (lcf_features, self.opt.eta * left_lcf_features, (1 - self.opt.eta) * right_lcf_features), -1)
            else:
                cat_features = torch.cat((lcf_features, left_lcf_features, right_lcf_features), -1)
            sent_out = self.linear_window_3h(cat_features)
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, left_lcf_features), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, right_lcf_features), -1))
        else:
            raise KeyError('Invalid parameter:', self.opt.window)

        sent_out = torch.cat((global_context_features, sent_out), -1)
        sent_out = self.post_linear(sent_out)
        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        dense_out = self.dense(sent_out)

        return {'logits': dense_out, 'hidden_state': sent_out}
