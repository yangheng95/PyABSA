# -*- coding: utf-8 -*-
# file: FAST_LSA_T_V2.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.lsa import LSA
from pyabsa.network.sa_encoder import Encoder


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
            # cdw_sent_out = self.CDW_LSA(global_context_features,
            #                             spc_mask_vec=spc_mask_vec,
            #                             lcf_matrix=lcf_cdw_matrix,
            #                             left_lcf_matrix=left_lcf_cdw_matrix,
            #                             right_lcf_matrix=right_lcf_cdw_matrix)
            # cdm_sent_out = self.CDM_LSA(global_context_features,
            #                             spc_mask_vec=spc_mask_vec,
            #                             lcf_matrix=lcf_cdm_matrix,
            #                             left_lcf_matrix=left_lcf_cdm_matrix,
            #                             right_lcf_matrix=right_lcf_cdm_matrix)
            # sent_out = self.fusion_linear(torch.cat((global_context_features, cdw_sent_out, cdm_sent_out), -1))
            sent_out = self.CDW_LSA(global_context_features,
                                    spc_mask_vec=spc_mask_vec,
                                    lcf_matrix=lcf_cdw_matrix,
                                    left_lcf_matrix=left_lcf_cdm_matrix,
                                    right_lcf_matrix=right_lcf_cdm_matrix)
            sent_out = torch.cat((global_context_features, sent_out), -1)
            sent_out = self.post_linear(sent_out)

        else:
            print('Invalid LCF mode: {}'.format(self.opt.lcf))
            sent_out = global_context_features

        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        dense_out = self.dense(sent_out)

        return {'logits': dense_out, 'hidden_state': sent_out}
