# -*- coding: utf-8 -*-
# file: generate_adversarial_examples.py
# time: 03/05/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import tqdm
from findfile import find_files
from metric_visualizer import MetricVisualizer

from termcolor import colored

from textattack import Attacker
from textattack.attack_recipes import (
    BERTAttackLi2020,
    BAEGarg2019,
    PWWSRen2019,
    TextFoolerJin2019,
    PSOZang2020,
    IGAWang2019,
    GeneticAlgorithmAlzantot2018,
    DeepWordBugGao2018,
)
from textattack.attack_results import SuccessfulAttackResult
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper

import os

from pyabsa.utils.pyabsa_utils import fprint


# Quiet TensorFlow.
def get_ensembled_tad_results(results):
    target_dict = {}
    for r in results:
        target_dict[r["label"]] = (
            target_dict.get(r["label"]) + 1 if r["label"] in target_dict else 1
        )

    return dict(zip(target_dict.values(), target_dict.keys()))[
        max(target_dict.values())
    ]
    # return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]


def get_ensembled_tc_results(results):
    target_dict = {}
    for r in results:
        target_dict[r["label"]] = (
            target_dict.get(r["label"]) + 1 if r["label"] in target_dict else 1
        )

    return dict(zip(target_dict.values(), target_dict.keys()))[
        max(target_dict.values())
    ]
    # return dict(zip(target_dict.values(), target_dict.keys()))[max(target_dict.values())]


if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# device = autocuda.auto_cuda()
# from textattack.augmentation import EasyDataAugmenter as Aug
#
# # Alter default values if desired
# eda_augmenter = Aug(pct_words_to_swap=0.3, transformations_per_example=2)

# import nlpaug.augmenter.word as naw
#
# bert_augmenter = naw.ContextualWordEmbsAug(
#     model_path='roberta-base', action="substitute", aug_p=0.3, device=autocuda.auto_cuda())

# raw_augs = augmenter.augment(text)


class PyABSAModelWrapper(HuggingFaceModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses
    like

        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

    We need to convert that to a format TextAttack understands, like

        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.predict(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs["probs"])
        return outputs


class SentAttacker:
    def __init__(self, model, recipe_class=BAEGarg2019):
        model = model
        model_wrapper = PyABSAModelWrapper(model)

        recipe = recipe_class.build(model_wrapper)
        # WordNet defaults to english. Set the default language to French ('fra')

        # recipe.transformation.language = "en"

        _dataset = [("", 0)]
        _dataset = Dataset(_dataset)

        self.attacker = Attacker(recipe, _dataset)


def adversarial_attack_detection_and_defense(dataset, attack_recipe):
    sent_attacker = SentAttacker(tad_classifier, attack_recipe)

    filter_key_words = [
        ".py",
        ".md",
        "readme",
        "log",
        "result",
        "zip",
        ".state_dict",
        ".model",
        ".png",
        "acc_",
        "f1_",
        ".origin",
        ".adv",
        ".csv",
        ".bak",
    ]

    dataset_file = {"train": [], "test": [], "valid": []}

    search_path = "./"
    task = "tc_datasets"
    dataset_file["train"] += find_files(
        search_path,
        [dataset, "train", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "test.", "synthesized"]
        + filter_key_words,
    )
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
    )
    dataset_file["valid"] += find_files(
        search_path,
        [dataset, "valid", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
    )
    dataset_file["valid"] += find_files(
        search_path,
        [dataset, "dev", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
    )

    for dat_type in [
        # 'train',
        # 'valid',
        "test"
    ]:
        data = []
        label_set = set()
        for data_file in dataset_file[dat_type]:
            fprint(colored("Attack: {}".format(data_file), "green"))

            with open(data_file, mode="r", encoding="utf8") as fin:
                lines = fin.readlines()
                for line in lines:
                    text, label = line.split("$LABEL$")
                    text = text.strip()
                    label = int(label.strip())
                    data.append((text, label))
                    label_set.add(label)

            all_num = 1e-10
            def_num = 1e-10
            acc_count = 0.0
            def_acc_count = 0.0
            det_acc_count = 0.0
            it = tqdm.tqdm(data[:300], desc="testing ")
            for text, label in it:
                result = sent_attacker.attacker.simple_attack(text, label)
                if isinstance(result, SuccessfulAttackResult):
                    infer_res = tad_classifier.predict(
                        result.perturbed_result.attacked_text.text
                        + "$LABEL${},{},{}".format(
                            result.original_result.ground_truth_output,
                            1,
                            result.perturbed_result.output,
                        ),
                        print_result=False,
                        defense="pwws",
                    )
                    def_num += 1
                    # if infer_res['pred_adv_tr_label'] == str(result.original_result.ground_truth_output):
                    #     def_acc_count += 1
                    # infer_res['label'] = infer_res['pred_adv_tr_label']

                    if infer_res["label"] == str(
                        result.original_result.ground_truth_output
                    ):
                        def_acc_count += 1
                    if infer_res["is_adv_label"] == "1":
                        det_acc_count += 1
                    pass
                else:
                    infer_res = tad_classifier.predict(
                        result.original_result.attacked_text.text
                        + "$LABEL${},{},{}".format(
                            result.original_result.ground_truth_output,
                            1,
                            result.perturbed_result.output,
                        ),
                        print_result=False,
                    )
                all_num += 1
                if infer_res["label"] == str(
                    result.original_result.ground_truth_output
                ):
                    acc_count += 1
                it.description = colored(
                    "Det Acc:{}|TAD Acc: {}|Res Acc: {}".format(
                        round(det_acc_count / def_num * 100, 2),
                        round(def_acc_count / def_num * 100, 2),
                        round(acc_count / all_num * 100, 2),
                    ),
                    "green",
                )
                it.update()
            mv.log_metric(
                attack_name,
                "Detection Accuracy",
                det_acc_count / def_num * 100,
            )
            mv.log_metric(
                attack_name,
                "Defense Accuracy",
                def_acc_count / def_num * 100,
            )
            mv.log_metric(attack_name, "Restored Accuracy", acc_count / all_num * 100)


if __name__ == "__main__":

    attack_name = "BAE"
    # attack_name = 'PWWS'
    # attack_name = 'TextFooler'

    # attack_name = 'PSO'Z
    # attack_name = 'IGA'
    # attack_name = 'WordBug'
    datasets = [
        # 'SST2',
        "Amazon",
        # 'agnews10k',
    ]
    from pyabsa import TextAdversarialDefense as TAD

    for dataset in datasets:
        tad_classifier = TAD.TADTextClassifier(
            # f'TAD-{dataset}{attack_name}',
            # f'TAD-{dataset}',
            # f'tadbert_{dataset}{attack_name}',
            f"tadbert_{dataset}",
            # auto_device=autocuda.auto_cuda()
            auto_device="cuda:0",
        )
        attack_recipes = {
            "bae": BAEGarg2019,
            "pwws": PWWSRen2019,
            "textfooler": TextFoolerJin2019,
            "pso": PSOZang2020,
            "iga": IGAWang2019,
            "GA": GeneticAlgorithmAlzantot2018,
            "wordbugger": DeepWordBugGao2018,
        }
        mv = MetricVisualizer(name="main_tad_bae")
        for _ in range(1):
            adversarial_attack_detection_and_defense(
                dataset, attack_recipe=attack_recipes[attack_name.lower()]
            )
        mv.summary("")
        mv.dump()
