# -*- coding: utf-8 -*-
# file: apc_instructor.py
# time: 2021/4/22 0022
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


import torch.nn as nn
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor


class APCTrainingInstructor(BaseTrainingInstructor):
    def _load_dataset_and_prepare_dataloader(self):
        raise NotImplementedError("Please implement this method in your subclass!")

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _train_and_evaluate(self, criterion):
        raise NotImplementedError("Please implement this method in your subclass!")

    def _k_fold_train_and_evaluate(self, criterion):
        raise NotImplementedError("Please implement this method in your subclass!")

    def _evaluate_acc_f1(self, test_dataloader):
        raise NotImplementedError("Please implement this method in your subclass!")

    def _init_misc(self):
        raise NotImplementedError("Please implement this method in your subclass!")

    def _cache_or_load_dataset(self):
        pass

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        return self._train(criterion)
