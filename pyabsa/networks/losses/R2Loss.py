# -*- coding: utf-8 -*-
# file: R2Loss.py
# time: 2022/11/24 20:06
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import torch
from torch import nn


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true, dim=0)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        r2 = 1 - ss_res / ss_tot
        return 1 - torch.mean(r2)
