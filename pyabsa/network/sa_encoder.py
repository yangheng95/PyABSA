# -*- coding: utf-8 -*-
# file: sa_encoder.py
# time: 2021/6/6 0006
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import torch
import torch.nn as nn
import numpy as np
from transformers.models.bert.modeling_bert import BertSelfAttention


class Encoder(nn.Module):
    def __init__(self, config, opt, layer_num=1):
        super(Encoder, self).__init__()
        self.opt = opt
        self.config = config
        self.encoder = nn.ModuleList([SelfAttention(config, opt) for _ in range(layer_num)])
        self.tanh = torch.nn.Tanh()

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
