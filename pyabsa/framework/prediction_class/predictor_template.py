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

from pyabsa.utils.text_utils.mlm import get_mlm_and_tokenizer
from torch import cuda

from pyabsa import TaskCodeOption
from pyabsa.framework.checkpoint_class.checkpoint_template import CheckpointManager


class InferenceModel:
    task_code = TaskCodeOption.Aspect_Polarity_Classification

    def __init__(self, checkpoint: Union[str, object] = None, config=None, **kwargs):
        """
        :param checkpoint: checkpoint path or checkpoint object
        :param kwargs:

        """

        self.cal_perplexity = kwargs.get("cal_perplexity", False)

        self.checkpoint = CheckpointManager().parse_checkpoint(
            checkpoint, task_code=self.task_code
        )

        self.config = config

        self.model = None
        self.dataset = None

    def to(self, device=None):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM") and self.MLM is not None:
            self.MLM.to(self.config.device)

    def cpu(self):
        self.config.device = "cpu"
        self.model.to("cpu")
        if hasattr(self, "MLM"):
            self.MLM.to("cpu")

    def cuda(self, device="cuda:0"):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM"):
            self.MLM.to(device)

    def __post_init__(self):
        self.config.label_to_index["-100"] = -100
        self.config.label_to_index[""] = -100
        self.config.index_to_label[-100] = ""

        self.infer_dataloader = None
        self.config.initializer = self.config.initializer

        if self.cal_perplexity:
            try:
                self.MLM, self.MLM_tokenizer = get_mlm_and_tokenizer(
                    self.model, self.config
                )
            except Exception as e:
                self.MLM, self.MLM_tokenizer = None, None

        self.to(self.config.device)

    def batch_predict(self, **kwargs):
        """
        Predict from a file of sentences.
        param: target_file: the file path of the sentences to be predicted.
        param: print_result: whether to print the result.
        param: save_result: whether to save the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        raise NotImplementedError("Please implement batch_infer() in your subclass!")

    def predict(self, **kwargs):

        """
        Predict from a sentence or a list of sentences.
        param: text: the sentence or a list of sentence to be predicted.
        param: print_result: whether to print the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        raise NotImplementedError("Please implement infer() in your subclass!")

    def _run_prediction(self, **kwargs):
        raise NotImplementedError("Please implement _infer() in your subclass!")

    def destroy(self):
        del self.model
        cuda.empty_cache()
        time.sleep(3)
