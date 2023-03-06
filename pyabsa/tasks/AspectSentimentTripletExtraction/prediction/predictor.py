# -*- coding: utf-8 -*-
# file: sentiment_classifier.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2020. All Rights Reserved.
import os
import pickle
from typing import Union

import torch
from findfile import find_file

from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from torch import nn
from tqdm import tqdm

from pyabsa import DeviceTypeOption

from pyabsa.utils.pyabsa_utils import fprint, set_device, print_args

from pyabsa.framework.flag_class import TaskCodeOption

from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.data_utils_for_inference import (
    ASTEInferenceDataset,
)
from pyabsa.tasks.AspectSentimentTripletExtraction.dataset_utils.aste_utils import (
    DataIterator,
    Metric,
)


class AspectSentimentTripletExtractor(InferenceModel):
    task_code = TaskCodeOption.Aspect_Sentiment_Triplet_Extraction

    def __init__(self, checkpoint=None, **kwargs):
        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

        # load from a trainer
        if self.checkpoint and not isinstance(self.checkpoint, str):
            fprint("Load sentiment classifier from trainer")
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            # load from a model path
            try:
                if "fine-tuned" in self.checkpoint:
                    raise ValueError(
                        "Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!"
                    )
                fprint("Load sentiment classifier from", self.checkpoint)

                state_dict_path = find_file(
                    self.checkpoint, ".state_dict", exclude_key=["__MACOSX"]
                )
                model_path = find_file(
                    self.checkpoint, ".model", exclude_key=["__MACOSX"]
                )
                tokenizer_path = find_file(
                    self.checkpoint, ".tokenizer", exclude_key=["__MACOSX"]
                )
                config_path = find_file(
                    self.checkpoint, ".config", exclude_key=["__MACOSX"]
                )

                fprint("config: {}".format(config_path))
                fprint("state_dict: {}".format(state_dict_path))
                fprint("model: {}".format(model_path))
                fprint("tokenizer: {}".format(tokenizer_path))

                with open(config_path, mode="rb") as f:
                    self.config = pickle.load(f)
                    self.config.auto_device = kwargs.get("auto_device", True)
                    set_device(self.config, self.config.auto_device)

                if state_dict_path or model_path:
                    if state_dict_path:
                        self.model = self.config.model(config=self.config).to(
                            self.config.device
                        )
                        self.model.load_state_dict(
                            torch.load(
                                state_dict_path, map_location=torch.device("cpu")
                            )
                        )
                    elif model_path:
                        self.model = torch.load(
                            model_path, map_location=DeviceTypeOption.CPU
                        )

                self.tokenizer = self.config.tokenizer

                if kwargs.get("verbose", False):
                    fprint("Config used in Training:")
                    print_args(self.config)

            except Exception as e:
                raise RuntimeError(
                    "Fail to load the model from {}! "
                    "Please make sure the version of checkpoint and PyABSA are compatible."
                    " Try to remove he checkpoint and download again"
                    " \nException: {} ".format(e, checkpoint)
                )

        self.dataset = ASTEInferenceDataset(self.config, self.tokenizer)

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
        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)

        save_path = os.path.join(
            os.getcwd(),
            "{}.{}.result.json".format(
                self.config.task_name, self.config.model.__name__
            ),
        )

        target_file = detect_infer_dataset(
            target_file, task_code=TaskCodeOption.Aspect_Sentiment_Triplet_Extraction
        )
        if not target_file:
            raise FileNotFoundError("Can not find inference datasets!")

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)

        return self._run_prediction(
            save_path=save_path if save_result else None, print_result=print_result
        )

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
        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)
        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError("Please specify your datasets path!")
        if isinstance(text, str):
            return self._run_prediction(print_result=print_result, **kwargs)[0]
        else:
            return self._run_prediction(print_result=print_result, **kwargs)

    def _run_prediction(self, save_path=None, print_result=True, **kwargs):
        self.model.eval()
        all_results = []
        with torch.no_grad():
            data_loader = DataIterator(
                self.dataset.convert_examples_to_features(), self.config
            )
            if len(self.dataset) > 1:
                it = tqdm(data_loader, desc="Predicting")
            else:
                it = data_loader
            for i, batch in enumerate(it):
                (
                    sentence_ids,
                    sentences,
                    token_ids,
                    lengths,
                    masks,
                    sens_lens,
                    token_ranges,
                    aspect_tags,
                    tags,
                    word_pair_position,
                    word_pair_deprel,
                    word_pair_pos,
                    word_pair_synpost,
                    tags_symmetry,
                ) = batch

                inputs = {
                    "token_ids": token_ids,
                    "masks": masks,
                    "word_pair_position": word_pair_position,
                    "word_pair_deprel": word_pair_deprel,
                    "word_pair_pos": word_pair_pos,
                    "word_pair_synpost": word_pair_synpost,
                }

                preds = self.model(inputs)[-1]
                preds = nn.functional.softmax(preds, dim=-1)
                preds = torch.argmax(preds, dim=3)

                metric = Metric(
                    self.config,
                    preds,
                    tags,
                    lengths,
                    sens_lens,
                    token_ranges,
                )

                new_result = {
                    "sentence_id": "",
                    "sentence": "",
                    "Triplets": [],
                    "True Triplets": [],
                }

                try:
                    results = metric.parse_triplet(golden=True)
                    for j, triplets in enumerate(results[0]):
                        for k, triplet in enumerate(triplets):
                            asp_head, asp_tail, opn_head, opn_tail, polarity = triplet
                            triplet = {
                                "Aspect": " ".join(
                                    sentences[j].split()[asp_head : asp_tail + 1]
                                ),
                                "Opinion": " ".join(
                                    sentences[j].split()[opn_head : opn_tail + 1]
                                ),
                                "Polarity": self.config.index_to_label[polarity],
                            }
                            new_result["True Triplets"].append(triplet)
                        all_results.append(new_result)
                except Exception as e:
                    results = metric.parse_triplet(golden=False)

                # Print results
                for j, triplets in enumerate(results[1]):
                    new_result["sentence_id"] = sentence_ids[j]
                    new_result["sentence"] = sentences[j]

                    for k, triplet in enumerate(triplets):
                        asp_head, asp_tail, opn_head, opn_tail, polarity = triplet
                        triplet = {
                            "Aspect": " ".join(
                                sentences[j].split()[asp_head : asp_tail + 1]
                            ),
                            "Opinion": " ".join(
                                sentences[j].split()[opn_head : opn_tail + 1]
                            ),
                            "Polarity": self.config.index_to_label[polarity],
                        }
                        new_result["Triplets"].append(triplet)
                    all_results.append(new_result)

            for result in all_results:
                fprint(result)

            return all_results

    def clear_input_samples(self):
        self.dataset.all_data = []


class Predictor(AspectSentimentTripletExtractor):
    pass
