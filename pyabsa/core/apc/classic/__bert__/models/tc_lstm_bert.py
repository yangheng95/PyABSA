# -*- coding: utf-8 -*-
# file: tc_lstm.py
# author: songyouwei <youwei0314@gmail.com>, ZhangYikai <zykhelloha@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from ..layers.dynamic_rnn import DynamicLSTM


class TC_LSTM_BERT(nn.Module):
    inputs = ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices']

    def __init__(self, bert, opt):
        super(TC_LSTM_BERT, self).__init__()
        self.embed = bert
        self.lstm_l = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        # Get the target and its length(target_len)
        x_l, x_r, target = inputs['left_with_aspect_indices'], inputs['right_with_aspect_indices'], inputs['aspect_indices']
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1, dtype=torch.float)[:, None, None]
        x_l, x_r, target = self.embed(x_l)['last_hidden_state'], self.embed(x_r)['last_hidden_state'], self.embed(target)['last_hidden_state']
        v_target = torch.div(target.sum(dim=1, keepdim=True),
                             target_len)  # v_{target} in paper: average the target words

        # the concatenation of word embedding and target vector v_{target}:
        x_l = torch.cat(
            (x_l, torch.cat(([v_target] * x_l.shape[1]), 1)),
            2
        )
        x_r = torch.cat(
            (x_r, torch.cat(([v_target] * x_r.shape[1]), 1)),
            2
        )

        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return {'logits': out}
