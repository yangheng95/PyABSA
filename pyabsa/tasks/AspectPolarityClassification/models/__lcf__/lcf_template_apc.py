# -*- coding: utf-8 -*-
# file: lcf_template_apc.py
# time: 2021/6/22
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import torch.nn as nn


class LCF_TEMPLATE_BERT(nn.Module):
    inputs = ["text_indices", "text_raw_bert_indices", "lcf_vec"]

    def __init__(self, bert, config):
        super(LCF_TEMPLATE_BERT, self).__init__()
        self.bert4global = bert
        self.bert4local = self.bert4global
        self.config = config
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        raise NotImplementedError(
            "This is a template ATEPC model based on LCF, "
            "please implement your model use this template."
        )
