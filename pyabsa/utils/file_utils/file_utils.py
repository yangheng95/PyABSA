# -*- coding: utf-8 -*-
# file: file_utils.py
# time: 2021/7/13 0020
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import json
import os
import pickle
import sys
import zipfile
from typing import Union, List

import numpy as np
import requests
import torch
import tqdm
from findfile import find_files, find_cwd_file
from termcolor import colored

from pyabsa.utils.pyabsa_utils import save_args, fprint


def meta_load(path, **kwargs):
    """
    Load data from a file, which can be plain text, json file, Excel file,
     pickle file, numpy file, torch file, pandas file, etc.
     File types: txt, json, pickle, npy, pkl, pt, torch, csv, xlsx, xls

    Args:
        path (str): The path to the file.
        kwargs: Other arguments for the corresponding load function.

    Returns:
        The loaded data.
    """
    # Ascii file
    if path.endswith(".txt"):
        return load_txt(path, **kwargs)
    # JSON file
    elif path.endswith(".json"):
        return load_json(path, **kwargs)
    elif path.endswith(".jsonl"):
        return load_jsonl(path, **kwargs)

    # Binary file
    elif path.endswith(".pickle") or path.endswith(".pkl"):
        return load_pickle(path, **kwargs)
    elif path.endswith(".npy"):
        return load_npy(path, **kwargs)
    elif path.endswith(".pt") or path.endswith(".torch"):
        return load_torch(path, **kwargs)
    elif path.endswith(".csv"):
        return load_csv(path, **kwargs)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return load_excel(path, **kwargs)
    else:
        return load_txt(path, **kwargs)


def meta_save(data, path, **kwargs):
    """
    Save data to a pickle file, which can be plain text, json file, Excel file,
     pickle file, numpy file, torch file, pandas file, etc.
     File types: txt, json, pickle, npy, pkl, pt, torch, csv, xlsx, xls

    Args:
        data: The data to be saved.
        path (str): The path to the file.
        kwargs: Other arguments for the corresponding save function.
    """

    # Ascii file
    if path.endswith(".txt"):
        save_txt(data, path, **kwargs)
    # JSON file
    elif path.endswith(".json"):
        save_json(data, path, **kwargs)
    elif path.endswith(".jsonl"):
        save_jsonl(data, path, **kwargs)

    # Binary file
    elif path.endswith(".pickle") or path.endswith(".pkl"):
        save_pickle(data, path, **kwargs)
    elif path.endswith(".npy"):
        save_npy(data, path, **kwargs)
    elif path.endswith(".pt") or path.endswith(".torch"):
        save_torch(data, path, **kwargs)
    elif path.endswith(".csv"):
        save_csv(data, path, **kwargs)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        save_excel(data, path, **kwargs)
    else:
        return save_txt(path, **kwargs)


def save_jsonl(data, file_path, **kwargs):
    """
    Save data to a jsonl file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def save_txt(data, file_path, **kwargs):
    """
    Save data to a plain text file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)


def save_json(data, file_path, **kwargs):
    """
    Save data to a json file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)


def save_excel(data, file_path, **kwargs):
    """
    Save data to an Excel file.
    """
    import pandas as pd

    data.to_excel(file_path, **kwargs)


def save_csv(data, file_path, **kwargs):
    """
    Save data to a csv file.
    """
    import pandas as pd

    data.to_csv(file_path, **kwargs)


def save_npy(data, file_path, **kwargs):
    """
    Save data to a numpy file.
    """
    np.save(file_path, data, **kwargs)


def save_torch(data, file_path, **kwargs):
    """
    Save data to a torch file.
    """
    with open(file_path, "wb") as f:
        torch.save(data, f, **kwargs)


def save_pickle(data, file_path, **kwargs):
    """
    Save data to a pickle file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(data, f, **kwargs)


def load_excel(file_path, **kwargs):
    """
    Load an Excel file and return the data.
    """
    import pandas as pd

    return pd.read_excel(file_path, **kwargs)


def load_csv(file_path, **kwargs):
    """
    Load a csv file and return the data.
    """
    import pandas as pd

    return pd.read_csv(file_path, **kwargs)


def load_npy(file_path, **kwargs):
    """
    Load a numpy file and return the data.
    """
    return np.load(file_path, **kwargs)


def load_torch(file_path, **kwargs):
    """
    Load a torch file and return the data.
    """
    with open(file_path, "rb") as f:
        return torch.load(f, **kwargs)


def load_pickle(file_path, **kwargs):
    """
    Load a pickle file and return the data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_txt(file_path):
    """
    Load a plain text file and return a list of strings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def remove_empty_line(files: Union[str, List[str]]):
    """
    Remove empty lines from the input files.
    """
    # Convert a single file path to a list of length 1 for convenience
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
    """
    Save a Python dictionary to a JSON file.
    """
    # Convert a string representation of a dictionary to a dictionary
    if isinstance(dic, str):
        dic = eval(dic)
    with open(save_path, "w", encoding="utf-8") as f:
        # Write the dictionary to the file
        str_ = json.dumps(dic, ensure_ascii=False)
        f.write(str_)


def load_json(file_path, **kwargs):
    """
    Load a JSON file and return a Python dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Read the file and parse the JSON string to a dictionary
        data = f.readline().strip()
        fprint(
            type(data), data
        )  # 'fprint' function is not defined, may need to be defined
        dic = json.loads(data)
    return dic


def load_jsonl(file_path, **kwargs):
    """
    Load a JSONL file and return a list of Python dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Read the file and parse the JSON string to a dictionary
        lines = f.readlines()
        dic_list = [json.loads(line.strip()) for line in lines]
    return dic_list


def load_dataset_from_file(fname, config):
    """
    Loads a dataset from one or multiple files.

    Args:
        fname (str or List[str]): The name of the file(s) containing the dataset.
        config (dict): The configuration dictionary containing the logger (optional) and the maximum number of data to load (optional).

    Returns:
        A list of strings containing the loaded dataset.

    Raises:
        ValueError: If an empty line is found in the dataset.

    """
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
    lines = lines[: config.get("data_num", None)]
    return lines


def prepare_glove840_embedding(glove_path, embedding_dim, config):
    """
    Check if the provided GloVe embedding exists, if not, search for a similar file in the current directory, or download
    the 840B GloVe embedding. If none of the above exists, raise an error.
    :param glove_path: str, path to the GloVe embedding
    :param embedding_dim: int, the dimension of the embedding
    :param config: dict, configuration dictionary
    :return: str, the path to the GloVe embedding
    """
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
    """
    Unzip a checkpoint file in zip format.

    Args:
        zip_path (str): path to the zip file.

    Returns:
        str: path to the unzipped checkpoint directory.

    """
    try:
        # Inform the user that the checkpoint file is being unzipped
        print("Find zipped checkpoint: {}, unzipping".format(zip_path))
        sys.stdout.flush()

        # Create a directory with the same name as the zip file to store the unzipped checkpoint files
        if not os.path.exists(zip_path):
            os.makedirs(zip_path.replace(".zip", ""))

        # Extract the contents of the zip file to the created directory
        z = zipfile.ZipFile(zip_path, "r")
        z.extractall(os.path.dirname(zip_path))

        # Inform the user that the unzipping is done
        print("Done.")
    except zipfile.BadZipfile:
        # If the zip file is corrupted, inform the user that the unzipping has failed
        print("{}: Unzip failed".format(zip_path))

    # Return the path to the unzipped checkpoint directory
    return zip_path.replace(".zip", "")


def save_model(config, model, tokenizer, save_path, **kwargs):
    """
    Save a trained model, configuration, and tokenizer to the specified path.

    Args:
        config (Config): Configuration for the model.
        model (nn.Module): The trained model.
        tokenizer: Tokenizer used by the model.
        save_path (str): The path where to save the model, config, and tokenizer.
        **kwargs: Additional keyword arguments.
    """
    if (
        hasattr(model, "module")
        or hasattr(model, "core")
        or hasattr(model, "_orig_mod")
    ):
        model_to_save = model.module
    else:
        model_to_save = model
    # Check the specified save mode.
    if config.save_mode == 1 or config.save_mode == 2:
        # Create save_path directory if it doesn't exist.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the configuration and tokenizer to the save_path directory.
        f_config = open(save_path + config.model_name + ".config", mode="wb")
        f_tokenizer = open(save_path + config.model_name + ".tokenizer", mode="wb")
        pickle.dump(config, f_config)
        pickle.dump(tokenizer, f_tokenizer)
        f_config.close()
        f_tokenizer.close()
        # Save the arguments used to create the configuration.
        save_args(config, save_path + config.model_name + ".args.txt")
        # Save the model state dict or the entire model depending on the save mode.
        if config.save_mode == 1:
            torch.save(
                model_to_save.state_dict(),
                save_path + config.model_name + ".state_dict",
            )
        elif config.save_mode == 2:
            torch.save(model.cpu(), save_path + config.model_name + ".model")

    elif config.save_mode == 3:
        # Save the fine-tuned BERT model.
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
