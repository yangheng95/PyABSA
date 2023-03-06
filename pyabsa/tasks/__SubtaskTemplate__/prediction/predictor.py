# -*- coding: utf-8 -*-
# file: sentiment_classifier.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2020. All Rights Reserved.
from abc import ABC
from typing import Union


from pyabsa.framework.prediction_class.predictor_template import InferenceModel


class AliasedClassifier(InferenceModel):
    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        """
        from_train_model: load inference model from trained model
        """

        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

        self.__post_init__(**kwargs)

    def batch_infer(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """
        A deprecated version of batch_predict method.

        Args:
            target_file (str): the path to the target file for inference
            print_result (bool): whether to print the result
            save_result (bool): whether to save the result
            ignore_error (bool): whether to ignore the error

        Returns:
            result (dict): a dictionary of the results
        """
        return self.batch_predict(
            target_file=target_file,
            print_result=print_result,
            save_result=save_result,
            ignore_error=ignore_error,
            **kwargs
        )

    def infer(self, text: str = None, print_result=True, ignore_error=True, **kwargs):
        """
        A deprecated version of the predict method.

        Args:
            text (str): the text to predict
            print_result (bool): whether to print the result
            ignore_error (bool): whether to ignore the error

        Returns:
            result (dict): a dictionary of the results
        """
        return self.predict(
            text=text, print_result=print_result, ignore_error=ignore_error, **kwargs
        )

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
        raise NotImplementedError("Please implement this method in your subtask class!")

    def predict(
        self,
        text: Union[str, list] = None,
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
        raise NotImplementedError("Please implement this method in your subtask class!")

    def _run_prediction(self, save_path=None, print_result=True, **kwargs):
        raise NotImplementedError("Please implement this method in your subtask class!")

    def clear_input_samples(self):
        self.dataset.all_data = []


# class Predictor(AliasedClassifier):
#     pass
