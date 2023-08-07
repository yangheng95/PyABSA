# -*- coding: utf-8 -*-
# file: text_classifier.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2020. All Rights Reserved.
import json
import os
import pickle
import time
from typing import Union

import torch
import tqdm
from findfile import find_file, find_cwd_dir
from termcolor import colored

from torch.utils.data import DataLoader
from transformers import AutoModel

from pyabsa import TaskCodeOption, DeviceTypeOption
from pyabsa.framework.prediction_class.predictor_template import InferenceModel
from ..dataset_utils.__classic__.data_utils_for_inference import (
    GloVeTADInferenceDataset,
)
from ..dataset_utils.__plm__.data_utils_for_inference import BERTTADInferenceDataset
from ..models import BERTTADModelList, GloVeTADModelList
from pyabsa.utils.data_utils.dataset_manager import detect_infer_dataset
from pyabsa.utils.pyabsa_utils import set_device, print_args, fprint


def init_attacker(tad_classifier, defense):
    try:
        from textattack import Attacker
        from textattack.attack_recipes import (
            BAEGarg2019,
            PWWSRen2019,
            TextFoolerJin2019,
            PSOZang2020,
            IGAWang2019,
            GeneticAlgorithmAlzantot2018,
            DeepWordBugGao2018,
        )
        from textattack.datasets import Dataset
        from textattack.models.wrappers import HuggingFaceModelWrapper

    except Exception as e:
        try:
            os.system("pip install git+https://github.com/yangheng95/TextAttack")
            from textattack import Attacker
            from textattack.attack_recipes import (
                BAEGarg2019,
                PWWSRen2019,
                TextFoolerJin2019,
                PSOZang2020,
                IGAWang2019,
                GeneticAlgorithmAlzantot2018,
                DeepWordBugGao2018,
            )
            from textattack.datasets import Dataset
            from textattack.models.wrappers import HuggingFaceModelWrapper
        except Exception as e:
            fprint(
                "If you need to evaluate text adversarial attack, please make sure you have installed:\n",
                colored(
                    "[1] pip install git+https://github.com/yangheng95/TextAttack\n",
                    "red",
                ),
                "and \n",
                colored("[2] pip install tensorflow_text \n", "red"),
            )
            fprint("Original error:", e)

    class PyABSAModelWrapper(HuggingFaceModelWrapper):
        def __init__(self, model):
            self.model = model  # pipeline = pipeline

        def __call__(self, text_inputs, **kwargs):
            outputs = []
            for text_input in text_inputs:
                raw_outputs = self.model.predict(
                    text_input, print_result=False, **kwargs
                )
                outputs.append(raw_outputs["probs"])
            return outputs

    class SentAttacker:
        def __init__(self, model, recipe_class=BAEGarg2019):
            model = model
            model_wrapper = PyABSAModelWrapper(model)

            recipe = recipe_class.build(model_wrapper)

            _dataset = [("", 0)]
            _dataset = Dataset(_dataset)

            self.attacker = Attacker(recipe, _dataset)

    attackers = {
        "bae": BAEGarg2019,
        "pwws": PWWSRen2019,
        "textfooler": TextFoolerJin2019,
        "pso": PSOZang2020,
        "iga": IGAWang2019,
        "ga": GeneticAlgorithmAlzantot2018,
        "wordbugger": DeepWordBugGao2018,
    }
    return SentAttacker(tad_classifier, attackers[defense])


class TADTextClassifier(InferenceModel):
    task_code = TaskCodeOption.Text_Adversarial_Defense

    def __init__(self, checkpoint=None, cal_perplexity=False, **kwargs):
        """
        from_train_model: load inference model from trained model
        """

        super().__init__(checkpoint, cal_perplexity, **kwargs)

        self.infer_dataloader = None
        self.cal_perplexity = cal_perplexity
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
                    if hasattr(BERTTADModelList, self.config.model.__name__):
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
                GloVeTADModelList, self.config.model.__name__
            ) and not hasattr(BERTTADModelList, self.config.model.__name__):
                raise KeyError(
                    "The checkpoint you are loading is not from classifier model."
                )

        if hasattr(BERTTADModelList, self.config.model.__name__):
            self.dataset = BERTTADInferenceDataset(
                config=self.config, tokenizer=self.tokenizer
            )
        else:
            self.dataset = GloVeTADInferenceDataset(
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
        """
        Batch prediction on an input file.
        :param target_file: the path of the input file
        :param print_result: whether to print the prediction results to the console
        :param save_result: whether to save the prediction results to a file
        :param ignore_error: whether to ignore errors during inference
        :param defense: adversarial defense technique to use during inference
        """
        return self.batch_predict(
            target_file=target_file,
            print_result=print_result,
            save_result=save_result,
            defense=defense,
            ignore_error=ignore_error,
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
        """
        Perform prediction on a single text or a list of texts.
        :param text: the text(s) to perform prediction on
        :param print_result: whether to print the prediction results to the console
        :param ignore_error: whether to ignore errors during inference
        :param defense: adversarial defense technique to use during inference
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
        defense: str = None,
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
            target_file, task_code=TaskCodeOption.Text_Adversarial_Defense
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
            save_path=save_path if save_result else None,
            print_result=print_result,
            defense=defense,
        )

    def predict(
        self,
        text: Union[str, list] = None,
        print_result=True,
        ignore_error=True,
        defense: str = None,
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
            return self._run_prediction(print_result=print_result, defense=defense)[0]
        else:
            return self._run_prediction(print_result=print_result, defense=defense)

    def _run_prediction(self, save_path=None, print_result=True, defense=None):
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        correct = {True: "Correct", False: "Wrong"}
        results = []

        with torch.no_grad():
            self.model.eval()
            n_correct = 0
            n_labeled = 0

            n_advdet_correct = 0
            n_advdet_labeled = 0
            if len(self.infer_dataloader.dataset) >= 100:
                it = tqdm.tqdm(self.infer_dataloader, desc="run inference")
            else:
                it = self.infer_dataloader
            for _, sample in enumerate(it):
                inputs = [
                    sample[col].to(self.config.device)
                    for col in self.config.inputs_cols
                ]
                outputs = self.model(inputs)
                logits, advdet_logits, adv_tr_logits = (
                    outputs["sent_logits"],
                    outputs["advdet_logits"],
                    outputs["adv_tr_logits"],
                )
                probs, advdet_probs, adv_tr_probs = (
                    torch.softmax(logits, dim=-1),
                    torch.softmax(advdet_logits, dim=-1),
                    torch.softmax(adv_tr_logits, dim=-1),
                )

                for i, (prob, advdet_prob, adv_tr_prob) in enumerate(
                    zip(probs, advdet_probs, adv_tr_probs)
                ):
                    text_raw = sample["text_raw"][i]

                    pred_label = int(prob.argmax(axis=-1))
                    pred_is_adv_label = int(advdet_prob.argmax(axis=-1))
                    pred_adv_tr_label = int(adv_tr_prob.argmax(axis=-1))
                    ref_label = (
                        int(sample["label"][i])
                        if int(sample["label"][i]) in self.config.index_to_label
                        else ""
                    )
                    ref_is_adv_label = (
                        int(sample["is_adv"][i])
                        if int(sample["is_adv"][i]) in self.config.index_to_is_adv
                        else ""
                    )
                    ref_adv_tr_label = (
                        int(sample["adv_train_label"][i])
                        if int(sample["adv_train_label"][i])
                        in self.config.index_to_adv_train_label
                        else ""
                    )

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

                    result = {
                        "text": text_raw,
                        "label": self.config.index_to_label[pred_label],
                        "probs": prob.cpu().numpy(),
                        "confidence": float(max(prob)),
                        "ref_label": self.config.index_to_label[ref_label]
                        if isinstance(ref_label, int)
                        else ref_label,
                        "ref_label_check": correct[pred_label == ref_label]
                        if ref_label != -100
                        else "",
                        "is_fixed": False,
                        "is_adv_label": self.config.index_to_is_adv[pred_is_adv_label],
                        "is_adv_probs": advdet_prob.cpu().numpy(),
                        "is_adv_confidence": float(max(advdet_prob)),
                        "ref_is_adv_label": self.config.index_to_is_adv[
                            ref_is_adv_label
                        ]
                        if isinstance(ref_is_adv_label, int)
                        else ref_is_adv_label,
                        "ref_is_adv_check": correct[
                            pred_is_adv_label == ref_is_adv_label
                        ]
                        if ref_is_adv_label != -100
                        and isinstance(ref_is_adv_label, int)
                        else "",
                        "pred_adv_tr_label": self.config.index_to_label[
                            pred_adv_tr_label
                        ],
                        "ref_adv_tr_label": self.config.index_to_label[
                            ref_adv_tr_label
                        ],
                        "perplexity": perplexity,
                    }
                    if defense:
                        try:
                            if not hasattr(self, "sent_attacker"):
                                self.sent_attacker = init_attacker(
                                    self, defense.lower()
                                )
                            # if result["is_adv_label"] == "1":
                            #     res = self.sent_attacker.attacker.simple_attack(
                            #         text_raw, int(result["label"])
                            #     )
                            #     new_infer_res = self.predict(
                            #         res.perturbed_result.attacked_text.text,
                            #         print_result=False,
                            #     )
                            #     result["perturbed_label"] = result["label"]
                            #     result["label"] = new_infer_res["label"]
                            #     result["probs"] = new_infer_res["probs"]
                            #     result["ref_label_check"] = (
                            #         correct[int(result["label"]) == ref_label]
                            #         if ref_label != -100
                            #         else ""
                            #     )
                            #     result[
                            #         "restored_text"
                            #     ] = res.perturbed_result.attacked_text.text
                            #     result["is_fixed"] = True
                            # else:
                            #     result["restored_text"] = ""
                            #     result["is_fixed"] = False
                            res = self.sent_attacker.attacker.simple_attack(
                                text_raw, int(result["label"])
                            )
                            new_infer_res = self.predict(
                                res.perturbed_result.attacked_text.text,
                                print_result=False,
                            )
                            result["perturbed_label"] = result["label"]
                            result["label"] = new_infer_res["label"]
                            result["probs"] = new_infer_res["probs"]
                            result["ref_label_check"] = (
                                correct[int(result["label"]) == ref_label]
                                if ref_label != -100
                                else ""
                            )
                            result[
                                "restored_text"
                            ] = res.perturbed_result.attacked_text.text
                            result["is_fixed"] = True
                        except Exception as e:
                            fprint(
                                "Error:{}, try install TextAttack and tensorflow_text after 10 seconds".format(
                                    e
                                )
                            )
                            time.sleep(10)
                            raise RuntimeError("Installation done, please run again")

                    if ref_label != -100:
                        n_labeled += 1

                        if result["label"] == result["ref_label"]:
                            n_correct += 1

                    if ref_is_adv_label != -100:
                        n_advdet_labeled += 1
                        if ref_is_adv_label == pred_is_adv_label:
                            n_advdet_correct += 1

                    results.append(result)

        try:
            if print_result:
                for ex_id, result in enumerate(results):
                    text_printing = result["text"][:]
                    text_info = ""
                    if result["label"] != "-100":
                        if not result["ref_label"]:
                            text_info += " -> <CLS:{}(ref:{} confidence:{})>".format(
                                result["label"],
                                result["ref_label"],
                                result["confidence"],
                            )
                        elif result["label"] == result["ref_label"]:
                            text_info += colored(
                                " -> <CLS:{}(ref:{} confidence:{})>".format(
                                    result["label"],
                                    result["ref_label"],
                                    result["confidence"],
                                ),
                                "green",
                            )
                        else:
                            text_info += colored(
                                " -> <CLS:{}(ref:{} confidence:{})>".format(
                                    result["label"],
                                    result["ref_label"],
                                    result["confidence"],
                                ),
                                "red",
                            )

                    # AdvDet
                    if result["is_adv_label"] != "-100":
                        if not result["ref_is_adv_label"]:
                            text_info += " -> <AdvDet:{}(ref:{} confidence:{})>".format(
                                result["is_adv_label"],
                                result["ref_is_adv_check"],
                                result["is_adv_confidence"],
                            )
                        elif result["is_adv_label"] == result["ref_is_adv_label"]:
                            text_info += colored(
                                " -> <AdvDet:{}(ref:{} confidence:{})>".format(
                                    result["is_adv_label"],
                                    result["ref_is_adv_label"],
                                    result["is_adv_confidence"],
                                ),
                                "green",
                            )
                        else:
                            text_info += colored(
                                " -> <AdvDet:{}(ref:{} confidence:{})>".format(
                                    result["is_adv_label"],
                                    result["ref_is_adv_label"],
                                    result["is_adv_confidence"],
                                ),
                                "red",
                            )
                    text_printing += text_info
                    if self.cal_perplexity:
                        text_printing += colored(
                            " --> <perplexity:{}>".format(result["perplexity"]),
                            "yellow",
                        )
                    fprint("Example {}: {}".format(ex_id, text_printing))
            if save_path:
                with open(save_path, "w", encoding="utf8") as fout:
                    json.dump(str(results), fout, ensure_ascii=False)
                    fprint("inference result saved in: {}".format(save_path))
        except Exception as e:
            fprint("Can not save result: {}, Exception: {}".format(text_raw, e))

        if len(results) > 1:
            fprint(
                "CLS Acc:{}%".format(100 * n_correct / n_labeled if n_labeled else "")
            )
            fprint(
                "AdvDet Acc:{}%".format(
                    100 * n_advdet_correct / n_advdet_labeled
                    if n_advdet_labeled
                    else ""
                )
            )

        return results

    def clear_input_samples(self):
        self.dataset.all_data = []


class Predictor(TADTextClassifier):
    pass
