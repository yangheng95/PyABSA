# -*- coding: utf-8 -*-
# file: search_adversarial_example.py
# time: 6:49 PM 6/1/2023
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.
import json

from findfile import find_files
from termcolor import colored

from pyabsa.utils.pyabsa_utils import fprint

from chatgpt import Chatbot

chatbot = Chatbot()
prompt = """I want you to be a professional prompt engineer. 
Now I am working on the adversarial attack for text classification, and I need your to understand the original text and 
transform it to inject perturbation to make the model misclassify the text.
Here I give you an example, please convert it into an adversarial example by paraphrasing:
>>>\n{}"""


def search_adversarial_example(dataset, tad_classifier):
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
        return_relative_path=False,
    )
    dataset_file["test"] += find_files(
        search_path,
        [dataset, "test", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
        return_relative_path=False,
    )
    dataset_file["valid"] += find_files(
        search_path,
        [dataset, "valid", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
        return_relative_path=False,
    )
    dataset_file["valid"] += find_files(
        search_path,
        [dataset, "dev", task],
        exclude_key=[".adv", ".org", ".defense", ".inference", "train.", "synthesized"]
        + filter_key_words,
        return_relative_path=False,
    )

    for dat_type in [
        # 'train',
        # 'valid',
        "test"
    ]:
        for data_file in dataset_file[dat_type]:
            fprint(colored("Attack: {}".format(data_file), "green"))

            with open(data_file, mode="r", encoding="utf8") as fin:
                with open(data_file + ".adv", mode="w", encoding="utf8") as fin_out:
                    lines = fin.readlines()
                    for line in lines:
                        text, label = line.split("$LABEL$")
                        text = text.strip()
                        label = label.strip()
                        ori_label = tad_classifier.predict(
                            text + "$LABEL${},{},{}".format(label, 1, label),
                            print_result=False,
                            defense=False,
                        )["label"]
                        if ori_label != label:
                            num_chance = 200
                            perturb_label = ori_label
                            while num_chance > 0 and str(ori_label) == str(
                                perturb_label
                            ):
                                num_chance -= 1
                                response = chatbot.chat(prompt=prompt.format(text))
                                perturb_label = tad_classifier.predict(
                                    response
                                    + "$LABEL${},{},{}".format(label, 1, label),
                                    print_result=False,
                                    defense=False,
                                )["label"]
                                if str(perturb_label).strip() != str(label).strip():
                                    fin_out.write(
                                        json.dumps(
                                            dict(
                                                text=text,
                                                adversarial_text=response,
                                                label=label,
                                                adversarial_label=perturb_label,
                                            )
                                        )
                                        + "\n"
                                    )
                                    print(
                                        json.dumps(
                                            dict(
                                                text=text,
                                                adversarial_text=response,
                                                label=label,
                                                adversarial_label=perturb_label,
                                            )
                                        )
                                        + "\n"
                                    )
                                    break


if __name__ == "__main__":
    datasets = [
        # 'SST2',
        # "amazon",
        "agnews10k",
    ]
    from pyabsa import TextAdversarialDefense as TAD

    for dataset in datasets:
        tad_classifier = TAD.TADTextClassifier(
            # f'TAD-{dataset}{attack_name}',
            f"TAD-{dataset}",
            # f'tadbert_{dataset}{attack_name}',
            # f"tadbert_{dataset}",
            # auto_device=autocuda.auto_cuda()
            auto_device="cuda:0",
        )
        search_adversarial_example(dataset, tad_classifier)
