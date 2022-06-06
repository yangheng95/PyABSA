# -*- coding: utf-8 -*-
# file: FAST_LSA_T_V2.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class LSA(nn.Module):
    def __init__(self, bert, opt):
        super(LSA, self).__init__()
        self.opt = opt

        self.encoder = Encoder(bert.config, opt)
        self.encoder_left = Encoder(bert.config, opt)
        self.encoder_right = Encoder(bert.config, opt)
        self.linear_window_3h = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.eta1 = nn.Parameter(torch.tensor(self.opt.eta, dtype=torch.float))
        self.eta2 = nn.Parameter(torch.tensor(self.opt.eta, dtype=torch.float))

    def forward(self, global_context_features, spc_mask_vec, lcf_matrix, left_lcf_matrix, right_lcf_matrix):
        masked_global_context_features = torch.mul(spc_mask_vec, global_context_features)

        # # --------------------------------------------------- #
        lcf_features = torch.mul(global_context_features, lcf_matrix)
        lcf_features = self.encoder(lcf_features)
        # # --------------------------------------------------- #
        left_lcf_features = torch.mul(masked_global_context_features, left_lcf_matrix)
        left_lcf_features = self.encoder_left(left_lcf_features)
        # # --------------------------------------------------- #
        right_lcf_features = torch.mul(masked_global_context_features, right_lcf_matrix)
        right_lcf_features = self.encoder_right(right_lcf_features)
        # # --------------------------------------------------- #
        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            if self.eta1 <= 0 and self.opt.eta != -1:
                torch.nn.init.uniform_(self.eta1)
                print('reset eta1 to: {}'.format(self.eta1.item()))
            if self.eta2 <= 0 and self.opt.eta != -1:
                torch.nn.init.uniform_(self.eta2)
                print('reset eta2 to: {}'.format(self.eta2.item()))
            if self.opt.eta >= 0:
                cat_features = torch.cat((lcf_features, self.eta1 * left_lcf_features, self.eta2 * right_lcf_features), -1)
            else:
                cat_features = torch.cat((lcf_features, left_lcf_features, right_lcf_features), -1)
            sent_out = self.linear_window_3h(cat_features)
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, self.eta1 * left_lcf_features), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, self.eta2 * right_lcf_features), -1))
        else:
            raise KeyError('Invalid parameter:', self.opt.window)

        return sent_out


class FAST_LSA_T_V2(nn.Module):
    inputs = ['text_bert_indices', 'spc_mask_vec',
              'lcf_cdw_vec', 'left_lcf_cdw_vec', 'right_lcf_cdw_vec',
              'lcf_cdm_vec', 'left_lcf_cdm_vec', 'right_lcf_cdm_vec',
              ]

    def __init__(self, bert, opt):
        super(FAST_LSA_T_V2, self).__init__()
        self.bert4global = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)

        self.post_encoder = Encoder(bert.config, opt)
        self.post_encoder_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)

        if self.opt.lcf == 'cdw':
            self.CDW_LSA = LSA(self.bert4global, self.opt)
        if self.opt.lcf == 'cdm':
            self.CDM_LSA = LSA(self.bert4global, self.opt)
        elif self.opt.lcf == 'fusion':
            self.CDW_LSA = LSA(self.bert4global, self.opt)
            self.CDM_LSA = LSA(self.bert4global, self.opt)
            self.fusion_linear = nn.Linear(opt.embed_dim * 3, opt.embed_dim)

        self.post_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs['text_bert_indices']
        spc_mask_vec = inputs['spc_mask_vec'].unsqueeze(2)
        lcf_cdw_matrix = inputs['lcf_cdw_vec'].unsqueeze(2)
        left_lcf_cdw_matrix = inputs['left_lcf_cdw_vec'].unsqueeze(2)
        right_lcf_cdw_matrix = inputs['right_lcf_cdw_vec'].unsqueeze(2)
        lcf_cdm_matrix = inputs['lcf_cdm_vec'].unsqueeze(2)
        left_lcf_cdm_matrix = inputs['left_lcf_cdm_vec'].unsqueeze(2)
        right_lcf_cdm_matrix = inputs['right_lcf_cdm_vec'].unsqueeze(2)

        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']

        if self.opt.lcf == 'cdw':
            sent_out = self.CDW_LSA(global_context_features,
                                    spc_mask_vec=spc_mask_vec,
                                    lcf_matrix=lcf_cdw_matrix,
                                    left_lcf_matrix=left_lcf_cdw_matrix,
                                    right_lcf_matrix=right_lcf_cdw_matrix)
            sent_out = torch.cat((global_context_features, sent_out), -1)
            sent_out = self.post_linear(sent_out)

        elif self.opt.lcf == 'cdm':
            sent_out = self.CDM_LSA(global_context_features,
                                    spc_mask_vec=spc_mask_vec,
                                    lcf_matrix=lcf_cdm_matrix,
                                    left_lcf_matrix=left_lcf_cdm_matrix,
                                    right_lcf_matrix=right_lcf_cdm_matrix)
            sent_out = torch.cat((global_context_features, sent_out), -1)
            sent_out = self.post_linear(sent_out)

        elif self.opt.lcf == 'fusion':
            cdw_sent_out = self.CDW_LSA(global_context_features,
                                        spc_mask_vec=spc_mask_vec,
                                        lcf_matrix=lcf_cdw_matrix,
                                        left_lcf_matrix=left_lcf_cdw_matrix,
                                        right_lcf_matrix=right_lcf_cdw_matrix)
            cdm_sent_out = self.CDM_LSA(global_context_features,
                                        spc_mask_vec=spc_mask_vec,
                                        lcf_matrix=lcf_cdm_matrix,
                                        left_lcf_matrix=left_lcf_cdm_matrix,
                                        right_lcf_matrix=right_lcf_cdm_matrix)
            sent_out = self.fusion_linear(torch.cat((global_context_features, cdw_sent_out, cdm_sent_out), -1))

        else:
            print('Invalid LCF mode: {}'.format(self.opt.lcf))
            sent_out = global_context_features

        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        dense_out = self.dense(sent_out)

        return {'logits': dense_out, 'hidden_state': sent_out}
