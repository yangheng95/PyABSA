# -*- coding: utf-8 -*-
# file: ClassImblanceCE.py
# time: 14:20 2022/12/23
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import numpy as np
import torch
from torch import nn


class ClassBalanceCrossEntropyLoss(nn.Module):
    r"""
    Reference:
    Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.

        Equation: Loss(x, c) = \frac{1-\beta}{1-\beta^{n_c}} * CrossEntropy(x, c)

    Class-balanced loss considers the real volumes, named effective numbers, of each class, \
    rather than nominal numeber of images provided by original datasets.

    Args:
        beta(float, double) : hyper-parameter for class balanced loss to control the cost-sensitive weights.
    """

    def __init__(self):
        super(ClassBalanceCrossEntropyLoss, self).__init__()
        self.beta = self.para_dict["cfg"].LOSS.ClassBalanceCE.BETA
        self.class_balanced_weight = np.array(
            [(1 - self.beta) / (1 - self.beta**N) for N in self.num_class_list]
        )
        self.class_balanced_weight = torch.FloatTensor(
            self.class_balanced_weight
            / np.sum(self.class_balanced_weight)
            * self.num_classes
        ).to(self.device)

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight
