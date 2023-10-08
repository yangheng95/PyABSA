# -*- coding: utf-8 -*-
# file: prediction_class.py
# time: 03/11/2022 13:22
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import time
from typing import Union

from torch import cuda

from pyabsa.framework.flag_class.flag_template import DeviceTypeOption
from pyabsa.utils.text_utils.mlm import get_mlm_and_tokenizer


class InferenceModel:
    task_code = None

    def __init__(self, checkpoint: Union[str, object] = None, config=None, **kwargs):
        """
        Initializes an instance of the InferenceModel class, used for performing inference on a trained model.

        :param checkpoint: checkpoint path or checkpoint object
        :param config: configuration object
        :param kwargs: additional keyword arguments
        """
        from pyabsa.framework.checkpoint_class.checkpoint_template import (
            CheckpointManager,
        )

        self.cal_perplexity = kwargs.get("cal_perplexity", False)

        # parse the provided checkpoint to obtain the checkpoint path and configuration
        self.checkpoint = CheckpointManager().parse_checkpoint(
            checkpoint, task_code=self.task_code
        )

        self.config = config

        self.model = None
        self.dataset = None

    def to(self, device=None):
        """
        Sets the device on which the model will perform inference.

        :param device: the device to use for inference
        """
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM") and self.MLM is not None:
            self.MLM.to(self.config.device)

    def cpu(self):
        """
        Sets the device to CPU for performing inference.
        """
        self.config.device = DeviceTypeOption.CPU
        self.model.to(DeviceTypeOption.CPU)
        if hasattr(self, "MLM"):
            self.MLM.to(DeviceTypeOption.CPU)

    def cuda(self, device="cuda:0"):
        """
        Sets the device to CUDA for performing inference.

        :param device: the CUDA device to use for inference
        """
        self.config.device = device
        self.model.to(device)
        if hasattr(self, "MLM"):
            self.MLM.to(device)

    def __post_init__(self, **kwargs):
        """
        Initializes the InferenceModel instance after its properties have been set.
        """
        for k, v in kwargs.items():
            self.config[k] = v

        try:
            # set default values for label indices
            self.config.label_to_index["-100"] = -100
            self.config.label_to_index[""] = -100
            self.config.index_to_label[-100] = ""
        except Exception as e:
            pass

        self.infer_dataloader = None

        # calculate perplexity if required
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
        """
        This method should be implemented in the subclass for running predictions using the trained model.

        :param kwargs: additional keyword arguments
        :return: predicted labels or other prediction outputs
        """
        raise NotImplementedError(
            "Please implement _run_prediction() in your subclass!"
        )

    def destroy(self):
        """
        Deletes the model from memory and empties the CUDA cache.
        """
        del self.model
        cuda.empty_cache()
        time.sleep(3)
