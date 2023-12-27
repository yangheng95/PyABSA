# -*- coding: utf-8 -*-
# file: sentiment_classifier.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2020. All Rights Reserved.
import os
import pickle
from typing import Union

import torch
from findfile import find_file
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyabsa.framework.flag_class import TaskCodeOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from pyabsa.tasks.UniversalSentimentAnalysis.dataset_utils.data_utils_for_inference import (
    USAInferenceDataset,
)
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import fprint, set_device


class USAPredictor(InferenceModel):
    task_code = TaskCodeOption.Universal_Sentiment_Analysis

    def __init__(self, checkpoint=None, **kwargs):
        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

        # load from a trainer
        if self.checkpoint and isinstance(self.checkpoint, str):
            fprint("Load sentiment classifier from trainer")
            try:
                fprint("Load text classifier from", self.checkpoint)
                state_dict_path = find_file(
                    self.checkpoint, key=".state_dict", exclude_key=["__MACOSX"]
                )
                model_path = find_file(
                    self.checkpoint, key=".model", exclude_key=["__MACOSX"]
                )
                tokenizer_path = find_file(
                    self.checkpoint, key=".tokenizer", exclude_key=["__MACOSX"]
                )
                config_path = find_file(
                    self.checkpoint, key=".config", exclude_key=["__MACOSX"]
                )

                fprint("config: {}".format(config_path))
                fprint("state_dict: {}".format(state_dict_path))
                fprint("model: {}".format(model_path))
                fprint("tokenizer: {}".format(tokenizer_path))

                with open(config_path, mode="rb") as f:
                    self.config = pickle.load(f)
                    self.config.auto_device = kwargs.get("auto_device", True)
                    set_device(self.config, self.config.auto_device)

                self.model = self.config.model(config=self.config)
                self.model.model.load_state_dict(
                    torch.load(state_dict_path, map_location="cpu"), strict=False
                )
                self.model.model.to(self.config.device)

            except Exception as e:
                raise RuntimeError(
                    "Fail to load the model from {}! "
                    "Please make sure the version of checkpoint and PyABSA are compatible."
                    " Try to remove he checkpoint and download again"
                    " \nException: {} ".format(checkpoint, e)
                )

        self.dataset = USAInferenceDataset(
            self.config, self.config.tokenizer, dataset_type="test"
        )

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
            self.dataset.prepare_infer_dataset(text, ignore_error=ignore_error)
        else:
            raise RuntimeError("Please specify your datasets path!")
        if isinstance(text, str):
            try:
                return self._run_prediction(print_result=print_result, **kwargs)[0]
            except Exception as e:
                return {
                    "text": text,
                    "output": None,
                    "error": str(e),
                    "error_type": "RuntimeError",
                }
        else:
            return self._run_prediction(print_result=print_result, **kwargs)

    def _run_prediction(self, save_path=None, print_result=True, **kwargs):
        self.model.model.eval()
        all_results = []
        with torch.no_grad():

            def collate_fn(batch):
                input_ids = [torch.tensor(example["input_ids"]) for example in batch]
                input_ids = pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.config.tokenizer.pad_token_id,
                )
                return input_ids

            dataloader = DataLoader(
                self.dataset.tokenized_dataset["test"],
                batch_size=self.config.batch_size,
                collate_fn=collate_fn,
            )
            predicted_output = []
            self.model.model.to(self.config.device)
            print("Model loaded to: ", self.config.device)

            for batch in tqdm(dataloader):
                batch = batch.to(self.config.device)
                output_ids = self.model.model.generate(batch)
                output_texts = self.config.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for output_text in output_texts:
                    predicted_output.append(
                        # self.config.usa_instructor.decode_input(output_text)
                        output_text
                    )

            return predicted_output

    def clear_input_samples(self):
        self.dataset.all_data = []


class Predictor(USAPredictor):
    pass
