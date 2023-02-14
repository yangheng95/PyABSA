# -*- coding: utf-8 -*-
# file: checkpoint_utils.py
# time: 02/11/2022 21:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import json
import os
from distutils.version import StrictVersion

import requests
import tqdm
from findfile import find_files, find_cwd_files, find_cwd_dir
from packaging import version
from pyabsa.framework.flag_class import TaskCodeOption
from termcolor import colored
from pyabsa import __version__ as current_version, PyABSAMaterialHostAddress
from pyabsa.utils.file_utils.file_utils import unzip_checkpoint
from pyabsa.utils.pyabsa_utils import fprint


def parse_checkpoint_info(t_checkpoint_map, task_code, show_ckpts=False):
    fprint(
        "*" * 10,
        colored(
            "Available {} model checkpoints for Version:{} (this version)".format(
                task_code, current_version
            ),
            "green",
        ),
        "*" * 10,
    )
    for i, checkpoint_name in enumerate(t_checkpoint_map):
        checkpoint = t_checkpoint_map[checkpoint_name]
        try:
            c_version = checkpoint["Available Version"]
        except:
            continue

        if "-" in c_version:
            min_ver, _, max_ver = c_version.partition("-")
        elif "+" in c_version:
            min_ver, _, max_ver = c_version.partition("-")
        else:
            min_ver = c_version
            max_ver = ""
        max_ver = max_ver if max_ver else "N.A."
        if max_ver == "N.A." or StrictVersion(min_ver) <= StrictVersion(
            current_version
        ) <= StrictVersion(max_ver):
            if show_ckpts:
                fprint("-" * 100)
                fprint("Checkpoint Name: {}".format(checkpoint_name))
                for key in checkpoint:
                    fprint("{}: {}".format(key, checkpoint[key]))
                fprint("-" * 100)
    return t_checkpoint_map


def available_checkpoints(task_code: TaskCodeOption = None, show_ckpts=False):
    """
    :param task_code: see TaskCodeOption, e.g.
        from pyabsa import TaskCodeOption
        TaskCodeOption.Aspect_Polarity_Classification
        TaskCodeOption.Aspect_Term_Extraction_and_Classification
        TaskCodeOption.Sentiment_Analysis
        TaskCodeOption.Text_Classification
        TaskCodeOption.Text_Adversarial_Defense
    :param show_ckpts: show all checkpoints
    """
    if task_code is None:
        fprint("Please specify the task code, e.g. from pyabsa import TaskCodeOption")
    try:  # from huggingface space
        checkpoint_url = PyABSAMaterialHostAddress + "raw/main/checkpoints-v2.0.json"
        response = requests.get(checkpoint_url)
        with open("./checkpoints-v2.0.json", "w") as f:
            json.dump(response.json(), f)
    except Exception as e:
        fprint(
            "Fail to download checkpoints info from huggingface space, try to download from local"
        )
    with open("./checkpoints-v2.0.json", "r", encoding="utf8") as f:
        checkpoint_map = json.load(f)

    t_checkpoint_map = {}
    for c_version in checkpoint_map:
        if "-" in c_version:
            min_ver, _, max_ver = c_version.partition("-")
        elif "+" in c_version:
            min_ver, _, max_ver = c_version.partition("+")
        else:
            min_ver = c_version
            max_ver = ""
        max_ver = max_ver if max_ver else "N.A."
        if max_ver == "N.A." or version.parse(min_ver) <= version.parse(
            current_version
        ) <= version.parse(max_ver):
            if task_code:
                t_checkpoint_map.update(
                    checkpoint_map[c_version][task_code.upper()]
                    if task_code.upper() in checkpoint_map[c_version]
                    else {}
                )
                parse_checkpoint_info(t_checkpoint_map, task_code, show_ckpts)

    return t_checkpoint_map if task_code else checkpoint_map


def download_checkpoint(task: str, language: str, checkpoint: dict):
    fprint(
        colored(
            "Notice: The pretrained model are used for testing, "
            "it is recommended to train the model on your own custom datasets",
            "red",
        )
    )
    huggingface_checkpoint_url = (
        PyABSAMaterialHostAddress
        + "resolve/main/checkpoints/{}/{}/{}".format(
            checkpoint["Language"], task.upper(), checkpoint["Checkpoint File"]
        )
    )

    tmp_dir = "{}_{}_CHECKPOINT".format(task.upper(), language.upper())
    dest_path = os.path.join("./checkpoints", tmp_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if (
        find_files(dest_path, ".model") or find_files(dest_path, ".state_dict")
    ) and find_files(dest_path, ".config"):
        fprint("Checkpoint already downloaded, skip")
        return dest_path

    if find_cwd_files(
        [
            checkpoint["Training Model"],
            checkpoint["Checkpoint File"].strip(".zip"),
            ".config",
        ]
    ):
        return find_cwd_dir(
            [checkpoint["Training Model"], checkpoint["Checkpoint File"].strip(".zip")]
        )
    save_path = os.path.join(dest_path, checkpoint["Checkpoint File"])

    try:  # from Huggingface Space
        response = requests.get(huggingface_checkpoint_url, stream=True)
        "https://huggingface.co/spaces/yangheng/PyABSA/raw/main/checkpoints/Code/bert_mlp_all_cpdp_acc_64.52_f1_64.48.zip"
        "https://huggingface.co/spaces/yangheng/PyABSA/resolve/main/checkpoints/Code/CDD/bert_mlp_all_cpdp_acc_64.52_f1_64.48.zip"
        with open(save_path, "wb") as f:
            for chunk in tqdm.tqdm(
                response.iter_content(chunk_size=1024 * 1024),
                unit="MB",
                total=int(response.headers["content-length"]) // 1024 // 1024,
                desc="Downloading checkpoint",
            ):
                f.write(chunk)
    except Exception as e:
        raise ConnectionError("Fail to download checkpoint: {}".format(e))
    unzip_checkpoint(save_path)
    os.remove(save_path)
    fprint(
        colored(
            "If the auto-downloading failed, please download it via browser: {} ".format(
                huggingface_checkpoint_url
            ),
            "yellow",
        )
    )
    return dest_path
