# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: yangheng <hy345@exeter.ac.uk>
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import random
from typing import Union

import numpy as np
import torch
import tqdm
from findfile import find_file, find_cwd_dir
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoModel

from sklearn import metrics

from pyabsa import TaskCodeOption, LabelPaddingOption, DeviceTypeOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from ..dataset_utils.__plm__.data_utils_for_inference import BERTTCInferenceDataset
from ..models import BERTTCModelList, GloVeTCModelList
from ..dataset_utils.__classic__.data_utils_for_inference import GloVeTCInferenceDataset
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import set_device, print_args, fprint, rprint
from pyabsa.utils.text_utils.mlm import get_mlm_and_tokenizer
from pyabsa.framework.tokenizer_class.tokenizer_class import (
    PretrainedTokenizer,
    Tokenizer,
    build_embedding_matrix,
)


class TextClassifier(InferenceModel):
    task_code = TaskCodeOption.Text_Classification

    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):

        """
        from_train_model: load inference model from trained model
        """

        super().__init__(checkpoint, cal_perplexity, task_code=self.task_code, **kwargs)

        # load from a trainer
        if self.checkpoint and not isinstance(self.checkpoint, str):
            fprint("Load text classifier from trainer")
            self.model = self.checkpoint[0]
            self.config = self.checkpoint[1]
            self.tokenizer = self.checkpoint[2]
        else:
            try:
                if "fine-tuned" in self.checkpoint:
                    raise ValueError(
                        "Do not support to directly load a fine-tuned model, please load a .state_dict or .model instead!"
                    )
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

                if state_dict_path or model_path:
                    if hasattr(BERTTCModelList, self.config.model.__name__):
                        if state_dict_path:
                            if kwargs.get("offline", False):
                                self.bert = AutoModel.from_pretrained(
                                    find_cwd_dir(
                                        self.config.pretrained_bert.split("/")[-1]
                                    )
                                )
                            else:
                                self.bert = AutoModel.from_pretrained(
                                    self.config.pretrained_bert
                                )
                            self.model = self.config.model(self.bert, self.config)
                            self.model.load_state_dict(
                                torch.load(
                                    state_dict_path, map_location=DeviceTypeOption.CPU
                                )
                            )
                        elif model_path:
                            self.model = torch.load(
                                model_path, map_location=DeviceTypeOption.CPU
                            )

                        try:
                            self.tokenizer = PretrainedTokenizer(self.config, **kwargs)
                        except ValueError:
                            if tokenizer_path:
                                with open(tokenizer_path, mode="rb") as f:
                                    self.tokenizer = pickle.load(f)
                    else:
                        self.embedding_matrix = self.config.embedding_matrix
                        self.tokenizer = self.config.tokenizer
                        if model_path:
                            self.model = torch.load(
                                model_path, map_location=DeviceTypeOption.CPU
                            )
                        else:
                            self.model = self.config.model(
                                self.embedding_matrix, self.config
                            ).to(self.config.device)
                            self.model.load_state_dict(
                                torch.load(
                                    state_dict_path, map_location=DeviceTypeOption.CPU
                                )
                            )

                if kwargs.get("verbose", False):
                    fprint("Config used in Training:")
                    print_args(self.config)

            except Exception as e:
                raise RuntimeError(
                    "Exception: {} Fail to load the model from {}! ".format(
                        e, self.checkpoint
                    )
                )

            if not hasattr(
                GloVeTCModelList, self.config.model.__name__
            ) and not hasattr(BERTTCModelList, self.config.model.__name__):
                raise KeyError(
                    "The checkpoint and PyABSA you are loading is not from classifier model."
                )

        if hasattr(BERTTCModelList, self.config.model.__name__):
            self.dataset = BERTTCInferenceDataset(
                config=self.config, tokenizer=self.tokenizer
            )

        elif hasattr(GloVeTCModelList, self.config.model.__name__):
            self.dataset = GloVeTCInferenceDataset(
                config=self.config, tokenizer=self.tokenizer
            )

        self.__post_init__()

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

    def _log_write_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        fprint(
            "n_trainable_params: {0}, n_nontrainable_params: {1}".format(
                n_trainable_params, n_nontrainable_params
            )
        )
        for arg in vars(self.config):
            if getattr(self.config, arg) is not None:
                fprint(">>> {0}: {1}".format(arg, getattr(self.config, arg)))

    def batch_infer(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):

        return self.batch_predict(
            target_file=target_file,
            print_result=print_result,
            save_result=save_result,
            ignore_error=ignore_error,
            **kwargs
        )

    def infer(
        self,
        text: Union[str, list] = None,
        print_result=True,
        ignore_error=True,
        **kwargs
    ):
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
        Predict from a file of sentences.
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
            target_file, task_code=TaskCodeOption.Text_Classification
        )
        if not target_file:
            raise FileNotFoundError("Can not find inference datasets!")

        self.dataset.prepare_infer_dataset(target_file, ignore_error=ignore_error)
        self.infer_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.eval_batch_size,
            pin_memory=True,
            shuffle=False,
        )
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
        Predict from a sentence or a list of sentences.
        param: text: the sentence or a list of sentence to be predicted.
        param: print_result: whether to print the result.
        param: ignore_error: whether to ignore the error when predicting.
        param: kwargs: other parameters.
        """
        self.config.eval_batch_size = kwargs.get("eval_batch_size", 32)
        self.infer_dataloader = DataLoader(
            dataset=self.dataset, batch_size=self.config.eval_batch_size, shuffle=False
        )
        if text:
            self.dataset.prepare_infer_sample(text, ignore_error=ignore_error)
        else:
            raise RuntimeError("Please specify your datasets path!")
        if isinstance(text, str):
            return self._run_prediction(print_result=print_result)[0]
        else:
            return self._run_prediction(print_result=print_result)

    def _run_prediction(self, save_path=None, print_result=True):

        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: "Correct", False: "Wrong"}
        results = []
        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0
            n_total = 0
            t_targets_all, t_outputs_all = None, None

            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, desc="run inference")
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = [
                    sample[col].to(self.config.device)
                    for col in self.config.inputs_cols
                    if col != "label"
                ]

                outputs = self.model(inputs)
                sen_logits = outputs
                t_probs = torch.softmax(sen_logits, dim=-1)

                if t_targets_all is None:
                    t_targets_all = np.array(
                        [
                            self.config.label_to_index[x]
                            if x in self.config.label_to_index
                            else LabelPaddingOption.SENTIMENT_PADDING
                            for x in sample["label"]
                        ]
                    )
                    t_outputs_all = np.array(sen_logits.cpu()).astype(np.float32)
                else:
                    t_targets_all = np.concatenate(
                        (
                            t_targets_all,
                            [
                                self.config.label_to_index[x]
                                if x in self.config.label_to_index
                                else LabelPaddingOption.SENTIMENT_PADDING
                                for x in sample["label"]
                            ],
                        ),
                        axis=0,
                    )
                    t_outputs_all = np.concatenate(
                        (t_outputs_all, np.array(sen_logits.cpu()).astype(np.float32)),
                        axis=0,
                    )

                for i, i_probs in enumerate(t_probs):
                    sent = self.config.index_to_label[int(i_probs.argmax(axis=-1))]
                    if sample["label"][i] != LabelPaddingOption.LABEL_PADDING:
                        real_sent = sample["label"][i]
                    else:
                        real_sent = "N.A."
                    if (
                        real_sent != LabelPaddingOption.LABEL_PADDING
                        and real_sent != str(LabelPaddingOption.LABEL_PADDING)
                    ):
                        n_labeled += 1

                    text_raw = sample["text_raw"][i]
                    ex_id = sample["ex_id"][i]

                    if self.cal_perplexity:
                        ids = self.MLM_tokenizer(
                            text_raw,
                            truncation=True,
                            padding="max_length",
                            max_length=self.config.max_seq_len,
                            return_tensors="pt",
                        )
                        ids["labels"] = ids["input_ids"].clone()
                        ids = ids.to(self.config.device)
                        loss = self.MLM(**ids)["loss"]
                        perplexity = float(torch.exp(loss / ids["input_ids"].size(1)))
                    else:
                        perplexity = "N.A."

                    results.append(
                        {
                            "ex_id": ex_id,
                            "text": text_raw,
                            "label": sent,
                            "confidence": float(max(i_probs)),
                            "probs": i_probs.cpu().numpy(),
                            "ref_label": real_sent,
                            "ref_check": correct[sent == real_sent]
                            if real_sent != str(LabelPaddingOption.LABEL_PADDING)
                            else "",
                            "perplexity": perplexity,
                        }
                    )
                    n_total += 1

        try:
            if print_result:
                for ex_id, result in enumerate(results):
                    text_printing = result["text"][:]
                    if result["ref_label"] != LabelPaddingOption.LABEL_PADDING:
                        if result["label"] == result["ref_label"]:
                            text_info = colored(
                                "#{}\t -> <{}(ref:{} confidence:{})>\t".format(
                                    result["ex_id"],
                                    result["label"],
                                    result["ref_label"],
                                    result["confidence"],
                                ),
                                "green",
                            )
                        else:
                            text_info = colored(
                                "#{}\t -> <{}(ref:{}) confidence:{}>\t".format(
                                    result["ex_id"],
                                    result["label"],
                                    result["ref_label"],
                                    result["confidence"],
                                ),
                                "red",
                            )
                    else:
                        text_info = "#{}\t -> {}\t".format(
                            result["ex_id"], result["label"]
                        )
                    if self.cal_perplexity:
                        text_printing += colored(
                            " --> <perplexity:{}>\t".format(result["perplexity"]),
                            "yellow",
                        )
                    text_printing = text_info + text_printing

                    fprint("Example {}".format(text_printing))
            if save_path:
                with open(save_path, "w", encoding="utf8") as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    fprint("inference result saved in: {}".format(save_path))
        except Exception as e:
            fprint("Can not save result: {}, Exception: {}".format(text_raw, e))

        if len(results) > 1:
            fprint("Total samples:{}".format(n_total))
            fprint("Labeled samples:{}".format(n_labeled))

            report = metrics.classification_report(
                t_targets_all,
                np.argmax(t_outputs_all, -1),
                digits=4,
                target_names=[
                    self.config.index_to_label[x] for x in self.config.index_to_label
                ],
            )
            fprint(
                "\n---------------------------- Classification Report ----------------------------\n"
            )
            rprint(report)
            fprint(
                "\n---------------------------- Classification Report ----------------------------\n"
            )

            report = metrics.confusion_matrix(
                t_targets_all,
                np.argmax(t_outputs_all, -1),
                labels=[
                    self.config.label_to_index[x] for x in self.config.label_to_index
                ],
            )
            fprint(
                "\n---------------------------- Confusion Matrix ----------------------------\n"
            )
            rprint(report)
            fprint(
                "\n---------------------------- Confusion Matrix ----------------------------\n"
            )

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []
