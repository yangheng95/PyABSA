# -*- coding: utf-8 -*-
# file: MAELoss.py
# time: 2022/11/24 20:11
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch
from torch import nn


class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))