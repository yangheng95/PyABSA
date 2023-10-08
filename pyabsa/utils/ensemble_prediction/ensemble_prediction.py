# -*- coding: utf-8 -*-
# file: ensemble_prediction.py
# time: 0:34 2022/12/15
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from typing import List

import numpy as np


class VoteEnsemblePredictor:
    def __init__(
        self,
        predictors: [List, dict],
        weights: [List, dict] = None,
        numeric_agg="average",
        str_agg="max_vote",
    ):
        """
        Initialize the VoteEnsemblePredictor.

        :param predictors: A list of checkpoints, or a dictionary of initialized predictors.
        :param weights: A list of weights for each predictor, or a dictionary of weights for each predictor.
        :param numeric_agg: The aggregation method for numeric data. Options are 'average', 'mean', 'max', 'min',
                            'median', 'mode', and 'sum'.
        :param str_agg: The aggregation method for string data. Options are 'max_vote', 'min_vote', 'vote', and 'mode'.
        """
        from pyabsa.tasks._Archive.RNAClassification import RNAClassifier
        from pyabsa.tasks.TextClassification import TextClassifier
        from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier

        if weights is not None:
            assert len(predictors) == len(
                weights
            ), "Checkpoints and weights should have the same length"
            assert type(predictors) == type(
                weights
            ), "Checkpoints and weights should have the same type"

        assert len(predictors) > 0, "Checkpoints should not be empty"

        numeric_agg_methods = {
            "average": np.mean,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "mode": lambda x: max(set(x), key=x.count),
            "sum": np.sum,
        }
        str_agg_methods = {
            "max_vote": lambda x: max(set(x), key=x.count),
            "min_vote": lambda x: min(set(x), key=x.count),
            "vote": lambda x: max(set(x), key=x.count),
            "mode": lambda x: max(set(x), key=x.count),
        }
        assert (
            numeric_agg in numeric_agg_methods
        ), "numeric_agg should be either: " + str(numeric_agg_methods.keys())
        assert str_agg in str_agg_methods, "str_agg should be either max or vote" + str(
            str_agg_methods.keys()
        )

        self.numeric_agg = numeric_agg_methods[numeric_agg]
        self.str_agg = str_agg_methods[str_agg]

        if isinstance(predictors, dict):
            self.predictors = predictors
            self.weights = list(weights.values()) if weights else [1] * len(predictors)
        elif isinstance(predictors, list):
            self.weights = weights if weights else [1] * len(predictors)

            try:
                self.predictors = {
                    ckpt: SentimentClassifier(checkpoint=ckpt) for ckpt in predictors
                }
            except Exception as e:
                pass

            try:
                self.predictors = {
                    ckpt: TextClassifier(checkpoint=ckpt) for ckpt in predictors
                }
            except Exception as e:
                pass

            try:
                self.predictors = {
                    ckpt: RNAClassifier(checkpoint=ckpt) for ckpt in predictors
                }
            except Exception as e:
                pass

    def __ensemble(self, result: dict):
        """
        Aggregate prediction results by calling the appropriate aggregation method.

        :param result: a dictionary containing the prediction results
        :return: the aggregated prediction result
        """
        if isinstance(result, dict):
            return self.__dict_aggregate(result)
        elif isinstance(result, list):
            return self.__list_aggregate(result)
        else:
            return result

    def __dict_aggregate(self, result: dict):
        """
        Recursively aggregate a dictionary of prediction results.

        :param result: a dictionary containing the prediction results
        :return: the aggregated prediction result
        """
        ensemble_result = {}
        for k, v in result.items():
            if isinstance(result[k], list):
                ensemble_result[k] = self.__list_aggregate(result[k])
            elif isinstance(result[k], dict):
                ensemble_result[k] = self.__dict_aggregate(result[k])
            else:
                ensemble_result[k] = result[k]
        return ensemble_result

    def __list_aggregate(self, result: list):
        if not isinstance(result, list):
            result = [result]

        assert all(
            isinstance(x, (type(result[0]))) for x in result
        ), "all type of result should be the same"

        if isinstance(result[0], list):
            for i, k in enumerate(result):
                result[i] = self.__list_aggregate(k)
            # start to aggregate
            try:
                new_result = self.numeric_agg(result)
            except Exception as e:
                try:
                    new_result = self.str_agg(result)
                except Exception as e:
                    new_result = result
            return [new_result]

        elif isinstance(result[0], dict):
            for k in result:
                result[k] = self.__dict_aggregate(result[k])
            return result

        # start to aggregate
        try:
            new_result = self.numeric_agg(result)
        except Exception as e:
            try:
                new_result = self.str_agg(result)
            except Exception as e:
                new_result = result

        return new_result

    def predict(self, text, ignore_error=False, print_result=False):
        """
        Predicts on a single text and returns the ensemble result.

        :param text: The text to perform prediction on
        :type text: str
        :param ignore_error: Whether to ignore any errors that occur during prediction, defaults to False
        :type ignore_error: bool
        :param print_result: Whether to print the prediction result, defaults to False
        :type print_result: bool
        :return: The ensemble prediction result
        :rtype: dict
        """
        # Initialize an empty dictionary to store the prediction result
        result = {}
        # Loop through each checkpoint and predictor in the ensemble
        for ckpt, predictor in self.predictors.items():
            # Perform prediction on the text using the predictor
            raw_result = predictor.predict(
                text, ignore_error=ignore_error, print_result=print_result
            )
            # For each key-value pair in the raw result dictionary
            for key, value in raw_result.items():
                # If the key is not already in the result dictionary
                if key not in result:
                    # Initialize an empty list for the key
                    result[key] = []
                # Append the value to the list the number of times specified by the corresponding weight
                for _ in range(self.weights[list(self.predictors.keys()).index(ckpt)]):
                    result[key].append(value)
        # Return the ensemble result by aggregating the values in the result dictionary
        return self.__ensemble(result)

    def batch_predict(self, texts, ignore_error=False, print_result=False):
        """
        Predicts on a batch of texts using the ensemble of predictors.
        :param texts: a list of strings to predict on.
        :param ignore_error: boolean indicating whether to ignore errors or raise exceptions when prediction fails.
        :param print_result: boolean indicating whether to print the raw results for each predictor.
        :return: a list of dictionaries, each dictionary containing the aggregated results of the corresponding text in the input list.
        """
        batch_raw_results = []
        for ckpt, predictor in self.predictors.items():
            from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier

            if isinstance(predictor, SentimentClassifier):
                raw_results = predictor.predict(
                    texts,
                    ignore_error=ignore_error,
                    print_result=print_result,
                    merge_results=False,
                )
            else:
                raw_results = predictor.predict(
                    texts, ignore_error=ignore_error, print_result=print_result
                )
            batch_raw_results.append(raw_results)

        batch_results = []
        for raw_result in batch_raw_results:
            for i, result in enumerate(raw_result):
                if i >= len(batch_results):
                    batch_results.append({})
                for key, value in result.items():
                    if key not in batch_results[i]:
                        batch_results[i][key] = []
                    for _ in range(
                        self.weights[list(self.predictors.keys()).index(ckpt)]
                    ):
                        batch_results[i][key].append(value)

        ensemble_results = []
        for result in batch_results:
            ensemble_results.append(self.__ensemble(result))
        return ensemble_results

    # def batch_predict(self, texts, ignore_error=False, print_result=False):
    #     batch_results = []
    #     for text in tqdm.tqdm(texts, desc='Batch predict: '):
    #         result = self.predict(text, ignore_error=ignore_error, print_result=print_result)
    #         batch_results.append(result)
    #     return batch_results
