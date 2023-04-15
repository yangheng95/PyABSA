# -*- coding: utf-8 -*-
# file: predictor_template.py
# time: 2022/11/21 19:44
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from typing import Union

from pyabsa.framework.flag_class.flag_template import DeviceTypeOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel


class Predictor(InferenceModel):
    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        """
        from_train_model: load inference model from trained model
        """
        raise NotImplementedError("Please implement this method in subclass")

    def to(self, device=None):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM"):
            self.MLM.to(self.config.device)

    def cpu(self):
        self.config.device = DeviceTypeOption.CPU
        self.model.to(DeviceTypeOption.CPU)
        if hasattr(self, "MLM"):
            self.MLM.to(DeviceTypeOption.CPU)

    def cuda(self, device="cuda:0"):
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM"):
            self.MLM.to(device)

    def batch_infer(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """
        Runs inference on a batch of data from a file or list, and returns the results.

        :param target_file: the path to a file containing the input data, or a list of input data
        :param print_result: whether to print the results to the console
        :param save_result: whether to save the results to a file
        :param ignore_error: whether to ignore errors during inference and continue with the remaining data
        :param kwargs: additional arguments to pass to the _run_prediction method
        :return: a list of results from running inference on the input data
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def infer(self, text: str = None, print_result=True, ignore_error=True, **kwargs):
        """
        Runs inference on a single input, and returns the result.

        :param text: the input text to run inference on
        :param print_result: whether to print the result to the console
        :param ignore_error: whether to ignore errors during inference and return None instead
        :param kwargs: additional arguments to pass to the _run_prediction method
        :return: the result from running inference on the input text
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def batch_predict(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """
        Predict the sentiment from a file of sentences.
        param: target_file: the file path of the sentences to be predicted.
        param: print_result: whether to print the result.
        param: save_result: whether to save the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def predict(
        self,
        text: Union[str, list[str]] = None,
        print_result=True,
        ignore_error=True,
        **kwargs
    ):
        """
        Predict the sentiment from a sentence or a list of sentences.
        param: text: the sentence to be predicted.
        param: print_result: whether to print the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def _run_prediction(self, save_path=None, print_result=True):
        """
        Run prediction on the data in the dataloader, update the results list and return the results.
         Args:
        save_path: path to save the results in a json file.
        print_result: if True, print the prediction results.
        """
        raise NotImplementedError("Please implement this method in your subclass!")

    def clear_input_samples(self):
        self.dataset.all_data = []
