# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
from typing import Union

import numpy as np
import torch
import tqdm
from findfile import find_file, find_cwd_dir
from termcolor import colored
from torch.utils.data import DataLoader
from transformers import AutoModel

from sklearn import metrics

from pyabsa.framework.flag_class.flag_template import (
    TaskCodeOption,
    LabelPaddingOption,
    DeviceTypeOption,
)
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from ..dataset_utils.__plm__.data_utils_for_inference import BERTTCInferenceDataset
from ..models import BERTTCModelList, GloVeTCModelList
from ..dataset_utils.__classic__.data_utils_for_inference import GloVeTCInferenceDataset
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import set_device, print_args, fprint, rprint


class TextClassifier(InferenceModel):
    """High-level predictor for Text Classification.

    Loads a trained text classification checkpoint (BERT-based or GloVe-based)
    and provides convenient inference APIs for single sentences and batch
    datasets. When gold labels are present, evaluation metrics are printed.
    """

    task_code = TaskCodeOption.Text_Classification

    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        """Initialize the text classifier from a trained checkpoint.

        Args:
            checkpoint: Path to a checkpoint directory or a tuple returned by
                the trainer (model, config, tokenizer).
            cal_perplexity: Whether to compute masked-LM perplexity for inputs
                if the underlying model/tokenizer support it.
            **kwargs: Optional keyword arguments such as `auto_device`,
                `offline`, and `verbose`.

        Raises:
            RuntimeError: If the checkpoint cannot be loaded.
            KeyError: If the checkpoint is incompatible with text classifiers.
        """

        super().__init__(checkpoint, task_code=self.task_code, **kwargs)

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
                                ),
                                strict=False,
                            )
                        elif model_path:
                            self.model = torch.load(
                                model_path, map_location=DeviceTypeOption.CPU
                            )

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

                self.tokenizer = self.config.tokenizer

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

        self.__post_init__(**kwargs)

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
        defense: str = None,
        **kwargs
    ):
        """Deprecated alias of `batch_predict`.

        Args:
            target_file: Path to the input file or directory.
            print_result: Whether to print formatted results.
            save_result: Whether to save JSON results.
            ignore_error: Skip malformed lines instead of raising errors.
            defense: Optional adversarial defense strategy.
            **kwargs: Additional inference options.

        Returns:
            List[dict]: Prediction results.
        """
        return self.batch_predict(
            target_file=target_file,
            print_result=print_result,
            save_result=save_result,
            ignore_error=ignore_error,
            defense=defense,
            **kwargs
        )

    def infer(
        self,
        text: Union[str, list] = None,
        print_result=True,
        ignore_error=True,
        defense: str = None,
        **kwargs
    ):
        """Deprecated alias of `predict` for single or multiple inputs.

        Args:
            text: A string or list of strings to infer.
            print_result: Whether to print formatted results.
            ignore_error: Skip malformed inputs instead of raising errors.
            defense: Optional adversarial defense strategy.
            **kwargs: Additional inference options.

        Returns:
            dict or List[dict]: Prediction results.
        """
        return self.predict(
            text=text,
            print_result=print_result,
            ignore_error=ignore_error,
            defense=defense,
            **kwargs
        )

    def batch_predict(
        self,
        target_file=None,
        print_result=True,
        save_result=False,
        ignore_error=True,
        **kwargs
    ):
        """Run text classification inference on a dataset file or directory.

        Args:
            target_file: Path to a file or directory containing inputs.
            print_result: Print formatted results to stdout.
            save_result: Save JSON results to the working directory.
            ignore_error: Skip malformed lines instead of raising errors.
            **kwargs: Additional options, e.g., `eval_batch_size`.

        Returns:
            List[dict]: Inference results.
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
        """Predict labels for a string or a list of strings.

        Args:
            text: Single text or a list of texts to classify.
            print_result: Print formatted results to stdout.
            ignore_error: Skip malformed inputs instead of raising errors.
            **kwargs: Additional options, e.g., `eval_batch_size`.

        Returns:
            dict or List[dict]: A single result for string input, otherwise a
            list of results.
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
        """Internal prediction loop for text classification.

        Executes the model over `self.infer_dataloader`, collects logits,
        computes predictions, and optionally prints and saves results. When
        reference labels are present, prints a classification report and
        confusion matrix.

        Args:
            save_path: Optional path to save JSON results.
            print_result: Whether to print formatted results to stdout.

        Returns:
            List[dict]: Inference results.
        """
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
                            (
                                self.config.label_to_index[x]
                                if x in self.config.label_to_index
                                else LabelPaddingOption.SENTIMENT_PADDING
                            )
                            for x in sample["label"]
                        ]
                    )
                    t_outputs_all = sen_logits.cpu().detach().numpy().astype(np.float32)
                else:
                    t_targets_all = np.concatenate(
                        (
                            t_targets_all,
                            [
                                (
                                    self.config.label_to_index[x]
                                    if x in self.config.label_to_index
                                    else LabelPaddingOption.SENTIMENT_PADDING
                                )
                                for x in sample["label"]
                            ],
                        ),
                        axis=0,
                    )
                    t_outputs_all = np.concatenate(
                        (t_outputs_all, sen_logits.cpu().detach().numpy().astype(np.float32)),
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
                            "ref_check": (
                                correct[sent == real_sent]
                                if real_sent != str(LabelPaddingOption.LABEL_PADDING)
                                else ""
                            ),
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
                    self.config.index_to_label[x]
                    for x in sorted(self.config.index_to_label.keys())
                    if x != -100
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
                    self.config.label_to_index[x]
                    for x in self.config.label_to_index
                    if x != "-100" and x != ""
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
        """Clear any previously prepared inference samples/dataset cache."""
        self.dataset.all_data = []


class Predictor(TextClassifier):
    pass
