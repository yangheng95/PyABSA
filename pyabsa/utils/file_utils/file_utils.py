# -*- coding: utf-8 -*-
# file: file_utils.py
# time: 2021/7/13 0020
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import json
import os
import pickle
import sys
import zipfile
from typing import Union, List

import requests
import torch
import tqdm
from findfile import find_files, find_cwd_file
from termcolor import colored

from pyabsa.utils.pyabsa_utils import save_args, fprint


def remove_empty_line(files: Union[str, List[str]]):
    if isinstance(files, str):
        files = [files]
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(file, "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip():
                    f.write(line)


def save_json(dic, save_path):
    if isinstance(dic, str):
        dic = eval(dic)
    with open(save_path, "w", encoding="utf-8") as f:
        # f.write(str(dict))
        str_ = json.dumps(dic, ensure_ascii=False)
        f.write(str_)


def load_json(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        data = f.readline().strip()
        fprint(type(data), data)
        dic = json.loads(data)
    return dic


def load_dataset_from_file(fname, config):
    logger = config.get("logger", None)
    lines = []
    if isinstance(fname, str):
        fname = [fname]

    for f in fname:
        if logger:
            logger.info("Load dataset from {}".format(f))
        else:
            fprint("Load dataset from {}".format(f))
        fin = open(f, "r", encoding="utf-8")
        _lines_ = fin.readlines()
        for i, line in enumerate(_lines_):
            if not line.strip():
                raise ValueError(
                    "empty line: #{} in {}, previous line: {}".format(
                        i, f, _lines_[i - 1]
                    )
                )
            lines.append(line.strip())
        fin.close()
    lines = lines[
        : config.get("data_num", -1) if config.get("data_num", -1) > 0 else None
    ]
    return lines


def prepare_glove840_embedding(glove_path, embedding_dim, config):
    if config.get("glove_or_word2vec_path", None):
        glove_path = config.glove_or_word2vec_path
        return glove_path

    logger = config.logger
    if os.path.exists(glove_path) and os.path.isfile(glove_path):
        return glove_path
    else:
        embedding_files = []
        dir_path = os.getenv("$HOME") if os.getenv("$HOME") else os.getcwd()

        if find_files(
            dir_path,
            ["glove", "B", "d", ".txt", str(embedding_dim)],
            exclude_key=".zip",
        ):
            embedding_files += find_files(
                dir_path, ["glove", "B", ".txt", str(embedding_dim)], exclude_key=".zip"
            )
        elif find_files(dir_path, ["word2vec", "d", ".txt"], exclude_key=".zip"):
            embedding_files += find_files(
                dir_path,
                ["word2vec", "d", ".txt", str(embedding_dim)],
                exclude_key=".zip",
            )
        else:
            embedding_files += find_files(
                dir_path, ["d", ".txt", str(embedding_dim)], exclude_key=".zip"
            )

        if embedding_files:
            logger.info(
                "Find embedding file: {}, use: {}".format(
                    embedding_files, embedding_files[0]
                )
            )
            return embedding_files[0]

        else:
            if config.embed_dim != 300:
                raise ValueError(
                    "Please provide embedding file for embedding dim: {} in current wording dir ".format(
                        config.embed_dim
                    )
                )
            zip_glove_path = os.path.join(
                os.path.dirname(glove_path), "glove.840B.300d.zip"
            )
            logger.info(
                "No GloVe embedding found at {},"
                " downloading glove.840B.300d.txt (2GB will be downloaded / 5.5GB after unzip)".format(
                    glove_path
                )
            )
            try:
                response = requests.get(
                    "https://huggingface.co/spaces/yangheng/PyABSA-ATEPC/resolve/main/open-access/glove.840B.300d.zip",
                    stream=True,
                )
                with open(zip_glove_path, "wb") as f:
                    for chunk in tqdm.tqdm(
                        response.iter_content(chunk_size=1024 * 1024),
                        unit="MB",
                        total=int(response.headers["content-length"]) // 1024 // 1024,
                        desc=colored("Downloading GloVe-840B embedding", "yellow"),
                    ):
                        f.write(chunk)
            except Exception as e:
                raise ValueError(
                    "Download failed, please download glove.840B.300d.zip from "
                    "https://nlp.stanford.edu/projects/glove/, unzip it and put it in {}.".format(
                        glove_path
                    )
                )

        if find_cwd_file("glove.840B.300d.zip"):
            logger.info("unzip glove.840B.300d.zip")
            with zipfile.ZipFile(find_cwd_file("glove.840B.300d.zip"), "r") as z:
                z.extractall()
            logger.info("Zip file extraction Done.")

        return prepare_glove840_embedding(glove_path, embedding_dim, config)


def unzip_checkpoint(zip_path):
    try:
        fprint("Find zipped checkpoint: {}, unzipping".format(zip_path))
        sys.stdout.flush()
        if not os.path.exists(zip_path):
            os.makedirs(zip_path.replace(".zip", ""))
        z = zipfile.ZipFile(zip_path, "r")
        z.extractall(os.path.dirname(zip_path))
        fprint("Done.")
    except zipfile.BadZipfile:
        fprint("{}: Unzip failed".format(zip_path))
    return zip_path.replace(".zip", "")


def save_model(config, model, tokenizer, save_path, **kwargs):
    # Save a trained model, configuration and tokenizer
    if (
        hasattr(model, "module")
        or hasattr(model, "core")
        or hasattr(model, "_orig_mod")
    ):
        # fprint("save model from data-parallel!")
        model_to_save = model.module
    else:
        # fprint("save a single cuda model!")
        model_to_save = model
    if config.save_mode == 1 or config.save_mode == 2:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f_config = open(save_path + config.model_name + ".config", mode="wb")
        f_tokenizer = open(save_path + config.model_name + ".tokenizer", mode="wb")
        pickle.dump(config, f_config)
        pickle.dump(tokenizer, f_tokenizer)
        f_config.close()
        f_tokenizer.close()
        save_args(config, save_path + config.model_name + ".args.txt")
        if config.save_mode == 1:
            torch.save(
                model_to_save.state_dict(),
                save_path + config.model_name + ".state_dict",
            )  # save the state dict
        elif config.save_mode == 2:
            torch.save(
                model.cpu(), save_path + config.model_name + ".model"
            )  # save the state dict

    elif config.save_mode == 3:
        # save the fine-tuned bert model
        model_output_dir = save_path + "fine-tuned-pretrained-model"
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        output_model_file = os.path.join(model_output_dir, "pytorch_model.bin")
        output_config_file = os.path.join(model_output_dir, "config.json")

        if hasattr(model_to_save, "bert4global"):
            model_to_save = model_to_save.bert4global
        elif hasattr(model_to_save, "bert"):
            model_to_save = model_to_save.bert

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        if hasattr(tokenizer, "tokenizer"):
            tokenizer.tokenizer.save_pretrained(model_output_dir)
        else:
            tokenizer.save_pretrained(model_output_dir)

    else:
        raise ValueError("Invalid save_mode: {}".format(config.save_mode))
    model.to(config.device)
