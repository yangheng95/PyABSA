# -*- coding: utf-8 -*-
# file: ssw_s.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder

# -*- coding: utf-8 -*-
# file: ssw_s.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPooler

from pyabsa.network.sa_encoder import Encoder


class SSW_T(nn.Module):
    inputs = ['text_bert_indices', 'spc_mask_vec', 'lcf_vec', 'left_lcf_vec', 'right_lcf_vec', 'polarity', 'left_dist', 'right_dist']

    def __init__(self, bert, opt):
        super(SSW_T, self).__init__()
        self.bert4global = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)

        self.encoder = Encoder(bert.config, opt)
        self.encoder_left = Encoder(bert.config, opt)
        self.encoder_right = Encoder(bert.config, opt)

        self.post_linear = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_window_3h = nn.Linear(opt.embed_dim * 3, opt.embed_dim)
        self.linear_window_2h = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        self.dist_embed = nn.Embedding(opt.max_seq_len, opt.embed_dim)

        self.post_encoder = Encoder(bert.config, opt)
        self.post_encoder_ = Encoder(bert.config, opt)
        self.bert_pooler = BertPooler(bert.config)

        self.linear_left_ = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.linear_right_ = nn.Linear(opt.embed_dim * 2, opt.embed_dim)

        self.classification_criterion = nn.CrossEntropyLoss()
        self.sent_dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs['text_bert_indices']
        spc_mask_vec = inputs['spc_mask_vec']
        lcf_matrix = inputs['lcf_vec'].unsqueeze(2)
        left_lcf_matrix = inputs['left_lcf_vec'].unsqueeze(2)
        right_lcf_matrix = inputs['right_lcf_vec'].unsqueeze(2)
        polarity = inputs['polarity'] if 'polarity' in inputs else None
        left_dist = self.dist_embed(inputs['left_dist'].unsqueeze(1))
        right_dist = self.dist_embed(inputs['right_dist'].unsqueeze(1))

        global_context_features = self.bert4global(text_bert_indices)['last_hidden_state']
        masked_global_context_features = torch.mul(spc_mask_vec, global_context_features)

        # # --------------------------------------------------- #
        lcf_features = torch.mul(masked_global_context_features, lcf_matrix)
        lcf_features = self.encoder(lcf_features)
        # # --------------------------------------------------- #
        left_lcf_features = torch.mul(masked_global_context_features, left_lcf_matrix)
        left_lcf_features = left_dist * self.encoder_left(left_lcf_features)
        # # --------------------------------------------------- #
        right_lcf_features = torch.mul(masked_global_context_features, right_lcf_matrix)
        right_lcf_features = right_dist * self.encoder_right(right_lcf_features)
        # # --------------------------------------------------- #

        if 'lr' == self.opt.window or 'rl' == self.opt.window:
            if self.opt.eta >= 0:
                cat_features = torch.cat(
                    (lcf_features, self.opt.eta * left_lcf_features, (1 - self.opt.eta) * right_lcf_features), -1)
            else:
                cat_features = torch.cat((left_lcf_features, lcf_features, right_lcf_features), -1)
            sent_out = self.linear_window_3h(cat_features)
        elif 'l' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, left_lcf_features), -1))
        elif 'r' == self.opt.window:
            sent_out = self.linear_window_2h(torch.cat((lcf_features, right_lcf_features), -1))
        else:
            sent_out = lcf_features

        sent_out = torch.cat((global_context_features, sent_out), -1)
        sent_out = self.post_linear(sent_out)
        sent_out = self.dropout(sent_out)
        sent_out = self.post_encoder_(sent_out)
        sent_out = self.bert_pooler(sent_out)
        sent_logits = self.sent_dense(sent_out)
        if polarity is not None:
            sent_loss = self.classification_criterion(sent_logits, polarity)
            return {'logits': sent_logits, 'hidden_state': sent_out, 'loss': sent_loss}
        else:
            return {'logits': sent_logits, 'hidden_state': sent_out}
