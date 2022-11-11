# -*- coding: utf-8 -*-
# file: prediction_class.py
# time: 03/11/2022 13:22
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import time
from typing import Union

from torch import cuda

from pyabsa import TaskCodeOption
from pyabsa.framework.checkpoint_class.checkpoint_template import CheckpointManager


class InferenceModel:
    task_code = TaskCodeOption.Aspect_Polarity_Classification

    def __init__(self, checkpoint: Union[str, object] = None,  config=None, **kwargs):
        '''
        :param checkpoint: checkpoint path or checkpoint object
        :param kwargs:

        '''

        self.cal_perplexity = kwargs.get('cal_perplexity', False)

        self.checkpoint = CheckpointManager().parse_checkpoint(checkpoint, task_code=self.task_code)

        self.config = config

        self.model = None
        self.dataset = None

    def to(self, device=None):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, 'MLM') and self.MLM is not None:
            self.MLM.to(self.config.device)

    def cpu(self):
        self.config.device = 'cpu'
        self.model.to('cpu')
        if hasattr(self, 'MLM'):
            self.MLM.to('cpu')

    def cuda(self, device='cuda:0'):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, 'MLM'):
            self.MLM.to(device)

    def batch_predict(self, **kwargs):

        raise NotImplementedError('Please implement batch_infer() in your subclass!')

    def predict(self, **kwargs):

        raise NotImplementedError('Please implement infer() in your subclass!')

    def _run_prediction(self, **kwargs):
        raise NotImplementedError('Please implement _infer() in your subclass!')

    def destroy(self):
        del self.model
        cuda.empty_cache()
        time.sleep(3)
