# -*- coding: utf-8 -*-
# file: hlcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn
import numpy as np
import copy

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class HLCF_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(HLCF_BERT, self).__init__()
        raise NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()
