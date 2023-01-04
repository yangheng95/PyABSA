# -*- coding: utf-8 -*-
# file: ensemble_prediction.py
# time: 0:34 2022/12/15
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from typing import List

import numpy as np
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier


class VoteEnsemblePredictor:
    def __init__(
        self,
        predictors: [List, dict],
        weights: [List, dict] = None,
        numeric_agg="average",
        str_agg="max_vote",
    ):
        """

        :param predictors: list of checkpoints, you can pass initialized predictors or pass the checkpoints
        e.g., checkpoints = [checkpoint1:predictor1, checkpoint2:predictor2， checkpoint3:predictor3 ...]
        :param weights: list of weights
        e.g., weights = [checkpoint1:weight1, checkpoint2:weight2， checkpoint3:weight3 ...]
        :param numeric_agg: aggregation method for numeric data, default is average
        :param str_agg: aggregation method for string data or other data, default is max
        """
        if weights is not None:
            assert len(predictors) == len(
                weights
            ), "Checkpoints and weights should have the same length"
            assert type(predictors) == type(
                weights
            ), "Checkpoints and weights should have the same type"
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
            self.checkpoints = list(predictors.keys())
            self.predictors = predictors
            self.weights = (
                list(weights.values()) if weights else [1] * len(self.checkpoints)
            )
        else:
            raise NotImplementedError(
                "Only support dict type for checkpoints and weights"
            )

    def __ensemble(self, result: dict):
        if isinstance(result, dict):
            return self.__dict_aggregate(result)
        elif isinstance(result, list):
            return self.__list_aggregate(result)
        else:
            return result

    def __dict_aggregate(self, result: dict):
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
        result = {}
        for ckpt, predictor in self.predictors.items():
            raw_result = predictor.predict(
                text, ignore_error=ignore_error, print_result=print_result
            )
            for key, value in raw_result.items():
                if key not in result:
                    result[key] = []
                for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                    result[key].append(value)
        return self.__ensemble(result)

    def batch_predict(self, texts, ignore_error=False, print_result=False):
        batch_raw_results = []
        for ckpt, predictor in self.predictors.items():
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
                    for _ in range(self.weights[self.checkpoints.index(ckpt)]):
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
