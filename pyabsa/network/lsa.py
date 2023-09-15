# -*- coding: utf-8 -*-
# file: local_sentiment_aggregation.py
# time: 06/06/2022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import torch
from pyabsa.network.sa_encoder import Encoder
from torch import nn


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

    def forward(
        self,
        global_context_features,
        spc_mask_vec,
        lcf_matrix,
        left_lcf_matrix,
        right_lcf_matrix,
    ):
        masked_global_context_features = torch.mul(
            spc_mask_vec, global_context_features
        )

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
        if "lr" == self.opt.window or "rl" == self.opt.window:
            if self.eta1 <= 0 and self.opt.eta != -1:
                torch.nn.init.uniform_(self.eta1)
                print("reset eta1 to: {}".format(self.eta1.item()))
            if self.eta2 <= 0 and self.opt.eta != -1:
                torch.nn.init.uniform_(self.eta2)
                print("reset eta2 to: {}".format(self.eta2.item()))
            if self.opt.eta >= 0:
                cat_features = torch.cat(
                    (
                        lcf_features,
                        self.eta1 * left_lcf_features,
                        self.eta2 * right_lcf_features,
                    ),
                    -1,
                )
            else:
                cat_features = torch.cat(
                    (lcf_features, left_lcf_features, right_lcf_features), -1
                )
            sent_out = self.linear_window_3h(cat_features)
        elif "l" == self.opt.window:
            sent_out = self.linear_window_2h(
                torch.cat((lcf_features, self.eta1 * left_lcf_features), -1)
            )
        elif "r" == self.opt.window:
            sent_out = self.linear_window_2h(
                torch.cat((lcf_features, self.eta2 * right_lcf_features), -1)
            )
        else:
            raise KeyError("Invalid parameter:", self.opt.window)

        return sent_out
