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


class VoteEnsemblePredictor:
    def __init__(self,
                 predictors: [List, dict],
                 weights: [List, dict] = None,
                 numeric_agg='average',
                 str_agg='max_vote',
                 **kwargs):
        """

        :param predictors: list of checkpoints, you can pass initialized predictors or pass the checkpoints
        e.g., checkpoints = [checkpoint1:predictor1, checkpoint2:predictor2， checkpoint3:predictor3 ...]
        :param weights: list of weights
        e.g., weights = [checkpoint1:weight1, checkpoint2:weight2， checkpoint3:weight3 ...]
        :param numeric_agg: aggregation method for numeric data, default is average
        :param str_agg: aggregation method for string data or other data, default is max
        """
        if weights is not None:
            assert len(predictors) == len(weights), 'Checkpoints and weights should have the same length'
            assert type(predictors) == type(weights), 'Checkpoints and weights should have the same type'
        numeric_agg_methods = {'average': np.mean, 'mean': np.mean,
                               'max': np.max, 'min': np.min, 'median': np.median,
                               'mode': lambda x: max(set(x), key=x.count), 'sum': np.sum}
        str_agg_methods = {'max_vote': lambda x: max(set(x), key=x.count),
                           'min_vote': lambda x: min(set(x), key=x.count),
                           'vote': lambda x: max(set(x), key=x.count),
                           'mode': lambda x: max(set(x), key=x.count),
                           }
        assert numeric_agg in numeric_agg_methods, 'numeric_agg should be either: ' + str(numeric_agg_methods.keys())
        assert str_agg in str_agg_methods, 'str_agg should be either max or vote' + str(str_agg_methods.keys())

        self.numeric_agg = numeric_agg_methods[numeric_agg]
        self.str_agg = str_agg_methods[str_agg]

        if isinstance(predictors, dict):
            self.checkpoints = list(predictors.keys())
            self.predictors = predictors
            self.weights = list(weights.values()) if weights else [1] * len(self.checkpoints)
        else:
            raise NotImplementedError('Only support dict type for checkpoints and weights')

    def predict(self, text, ignore_error=False, print_result=False):
        result = {}
        for ckpt, predictor in self.predictors.items():
            raw_result = predictor.predict(text, ignore_error=ignore_error, print_result=print_result)
            for key, value in raw_result.items():
                if key not in result:
                    result[key] = []
                for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                    result[key].append(value)
        ensemble_result = {}
        for key, value in result.items():
            ensemble_result[key] = value
            if not isinstance(value, list):
                value = [value]
            for i, v in enumerate(value):
                try:
                    if isinstance(ensemble_result[key][i], list):
                        ensemble_result[key][i] = self.numeric_agg(v)
                    else:
                        ensemble_result[key] = self.numeric_agg(value)
                except Exception as e:
                    try:
                        if isinstance(ensemble_result[key][i], list):
                            ensemble_result[key][i] = self.str_agg(v)
                        else:
                            ensemble_result[key] = self.str_agg(value)
                    except Exception as e:
                        ensemble_result[key] = value
        return ensemble_result

    # Test version
    # def batch_predict(self, texts, ignore_error=False, print_result=False):
    #     batch_results = []
    #     for ckpt, predictor in self.predictors.items():
    #         raw_results = predictor.predict(texts, ignore_error=ignore_error, print_result=print_result)
    #         for raw_result in raw_results:
    #             result = {}
    #             for key, value in raw_result.items():
    #                 if key not in result:
    #                     result[key] = []
    #                 for _ in range(self.weights[self.checkpoints.index(ckpt)]):
    #                     result[key].append(value)
    #             batch_results.append(result)
    #
    #     ensemble_results = []
    #     for result in batch_results:
    #         ensemble_result = {}
    #         for key, value in result.items():
    #             ensemble_result[key] = value
    #             if not isinstance(value, list):
    #                 value = [value]
    #             for i, v in enumerate(value):
    #                 try:
    #                     if isinstance(ensemble_result[key][i], list):
    #                         ensemble_result[key][i] = self.numeric_agg(v)
    #                     else:
    #                         ensemble_result[key] = self.numeric_agg(value)
    #                 except Exception as e:
    #                     try:
    #                         if isinstance(ensemble_result[key][i], list):
    #                             ensemble_result[key][i] = self.str_agg(v)
    #                         else:
    #                             ensemble_result[key] = self.str_agg(value)
    #                     except Exception as e:
    #                         ensemble_result[key] = value
    #         ensemble_results.append(ensemble_result)
    #     return ensemble_results

    def batch_predict(self, texts, ignore_error=False, print_result=False):
        batch_results = []
        for text in texts:
            result = self.predict(text, ignore_error=ignore_error, print_result=print_result)
            batch_results.append(result)
        return batch_results
