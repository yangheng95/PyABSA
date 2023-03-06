# -*- coding: utf-8 -*-
# file: LDMALoss.py
# time: 14:21 2022/12/23
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
import torch
from torch import nn


class LDAMLoss(nn.Module):
    """
    References:
    Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS 2019.

    Args:
        s(float, double) : the scale of logits, according to the official codes.
        max_m(float, double): margin on loss functions. See original paper's Equation (12) and (13)

    Notes: There are two hyper-parameters of LDAMLoss codes provided by official codes,
          but the authors only provided the settings on long-tailed CIFAR.
          Settings on other datasets are not avaliable (https://github.com/kaidic/LDAM-DRW/issues/5).
    """

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return self.cross_entropy(self.s * output, target, weight=self.weight)
