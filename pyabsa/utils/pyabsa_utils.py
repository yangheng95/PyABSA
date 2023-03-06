# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import sys
import time

import torch
from autocuda import auto_cuda, auto_cuda_name
from termcolor import colored

from pyabsa import __version__ as pyabsa_version
from pyabsa.framework.flag_class.flag_template import DeviceTypeOption


def save_args(config, save_path):
    """
    Save arguments to a file.

    Args:
    - config: A Namespace object containing the arguments.
    - save_path: A string representing the path of the file to be saved.

    Returns:
    None
    """
    f = open(os.path.join(save_path), mode="w", encoding="utf8")
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write("{}: {}\n".format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None):
    """
    Print the arguments to the console.

    Args:
    - config: A Namespace object containing the arguments.
    - logger: A logger object.

    Returns:
    None
    """
    args = [key for key in sorted(config.args.keys())]
    for arg in args:
        if arg != "dataset" and arg != "dataset_dict" and arg != "embedding_matrix":
            if logger:
                try:
                    logger.info(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], config.args_call_count[arg]
                        )
                    )
                except:
                    logger.info(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], 0
                        )
                    )
            else:
                try:
                    fprint(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], config.args_call_count[arg]
                        )
                    )
                except:
                    fprint(
                        "{0}:{1}\t-->\tCalling Count:{2}".format(
                            arg, config.args[arg], 0
                        )
                    )


def validate_absa_example(text: str, aspect: str, polarity: str, config):
    """
    Validate input text, aspect, and polarity to ensure they meet certain criteria.

    Args:
        - text (str): The input text to validate.
        - aspect (str): The input aspect to validate.
        - polarity (str): The input polarity to validate.
        - config: Configuration options.

    Returns:
    - warning (bool): Flag indicating whether there are any warnings.
    """

    # Ensure aspect is not longer than text
    if len(text) < len(aspect):
        raise ValueError(
            "AspectLengthExceedTextError -> <aspect: {}> is longer than <text: {}>, <polarity: {}>".format(
                aspect, text, polarity
            )
        )

    # Ensure aspect is in text
    if aspect.strip().lower() not in text.strip().lower():
        raise ValueError(
            "AspectNotInTextError -> <aspect: {}> is not in <text: {}>>".format(
                aspect, text
            )
        )

    warning = False

    # Raise a warning if aspect is too long
    if len(aspect.split(" ")) > 10:
        config.logger.warning(
            "AspectTooLongWarning -> <aspect: {}> is too long, <text: {}>, <polarity: {}>".format(
                aspect, text, polarity
            )
        )
        warning = True

    # Ensure polarity is not too long
    if len(polarity.split(" ")) > 3:
        config.logger.warning(
            "LabelTooLongWarning -> <polarity: {}> is too long, <text: {}>, <aspect: {}>".format(
                polarity, text, aspect
            )
        )
        warning = True

    # Ensure polarity is not null
    if not polarity.strip():
        raise ValueError(
            "PolarityIsNullError -> <text: {}>, <aspect: {}>, <polarity: {}>".format(
                aspect, text, polarity
            )
        )

    # Raise a warning if aspect equals text
    if text.strip() == aspect.strip():
        config.logger.warning(
            "AspectEqualsTextWarning -> <aspect: {}> equals <text: {}>, <polarity: {}>".format(
                aspect, text, polarity
            )
        )
        warning = True

    # Ensure text is not null
    if not text.strip():
        raise ValueError(
            "TextIsNullError -> <text: {}>, <aspect: {}>, <polarity: {}>".format(
                aspect, text, polarity
            )
        )

    return warning


def check_and_fix_labels(label_set: set, label_name, all_data, config):
    """
    Check and fix the labels of the dataset.

    Args:
        label_set (set): A set of unique labels in the dataset.
        label_name (str): Name of the label column in the dataset.
        all_data (list): List of dictionaries containing the dataset.
        config (Config): The config object.

    Returns:
        None.
    """
    if "-100" in label_set:
        # Create label_to_index and index_to_label dictionaries for mapping labels to their corresponding indices
        # If "-100" is in the label_set, then map "-100" to -100
        label_to_index = {
            origin_label: int(idx) - 1 if origin_label != "-100" else -100
            for origin_label, idx in zip(sorted(label_set), range(len(label_set)))
        }
        index_to_label = {
            int(idx) - 1 if origin_label != "-100" else -100: origin_label
            for origin_label, idx in zip(sorted(label_set), range(len(label_set)))
        }
    else:
        # Create label_to_index and index_to_label dictionaries for mapping labels to their corresponding indices
        label_to_index = {
            origin_label: int(idx)
            for origin_label, idx in zip(sorted(label_set), range(len(label_set)))
        }
        index_to_label = {
            int(idx): origin_label
            for origin_label, idx in zip(sorted(label_set), range(len(label_set)))
        }

    # Save label_to_index and index_to_label in the config object if not already saved
    if "index_to_label" not in config.args:
        config.index_to_label = index_to_label
        config.label_to_index = label_to_index

    # Update the label_to_index and index_to_label dictionaries in the config object if needed
    if config.index_to_label != index_to_label:
        config.index_to_label.update(index_to_label)
        config.label_to_index.update(label_to_index)

    # Count the number of labels in the dataset
    num_label = {label: 0 for label in label_set}
    num_label["Sum"] = len(all_data)
    for item in all_data:
        # Map the label to its corresponding index
        try:
            num_label[item[label_name]] += 1
            item[label_name] = label_to_index[item[label_name]]
        except Exception as e:
            num_label[item.polarity] += 1
            item.polarity = label_to_index[item.polarity]

    # Log the label distribution in the dataset
    config.logger.info("Dataset Label Details: {}".format(num_label))


def check_and_fix_IOB_labels(label_map, config):
    """
    Check and fix IOB labels.

    Args:
        label_map (dict): A dictionary that maps IOB labels to their corresponding indices.
        config (Config): A configuration object.

    Returns:
        None
    """
    index_to_IOB_label = {
        int(label_map[origin_label]): origin_label for origin_label in label_map
    }
    config.index_to_IOB_label = index_to_IOB_label


def set_device(config, auto_device):
    """
    Sets the device to be used for the PyTorch model.

    :param config: An instance of ConfigManager class that holds the configuration for the model.
    :param auto_device: Specifies the device to be used for the model. It can be either a string, a boolean, or None.
                        If it is a string, it can be either "cuda", "cuda:0", "cuda:1", or "cpu".
                        If it is a boolean and True, it automatically selects the available CUDA device.
                        If it is None, it uses the autocuda.
    :return: device: The device to be used for the PyTorch model.
             device_name: The name of the device.
    """
    device_name = "Unknown"
    if isinstance(auto_device, str) and auto_device == DeviceTypeOption.ALL_CUDA:
        device = "cuda"
    elif isinstance(auto_device, str):
        device = auto_device
    elif isinstance(auto_device, bool):
        device = auto_cuda() if auto_device else DeviceTypeOption.CPU
    else:
        device = auto_cuda()
        try:
            torch.device(device)
        except RuntimeError as e:
            print(
                colored("Device assignment error: {}, redirect to CPU".format(e), "red")
            )
            device = DeviceTypeOption.CPU
    if device != DeviceTypeOption.CPU:
        device_name = auto_cuda_name()
    config.device = device
    config.device_name = device_name
    fprint("Set Model Device: {}".format(device))
    fprint("Device Name: {}".format(device_name))
    return device, device_name


def fprint(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
    """
    Custom print function that adds a timestamp and the pyabsa version before the printed message.

    Args:
        *objects: Any number of objects to be printed
        sep (str, optional): Separator between objects. Defaults to " ".
        end (str, optional): Ending character after all objects are printed. Defaults to "\n".
        file (io.TextIOWrapper, optional): Text file to write printed output to. Defaults to sys.stdout.
        flush (bool, optional): Whether to flush output buffer after printing. Defaults to False.
    """
    print(
        time.strftime(
            "[%Y-%m-%d %H:%M:%S] ({})".format(pyabsa_version),
            time.localtime(time.time()),
        ),
        *objects,
        sep=sep,
        end=end,
        file=file,
        flush=flush
    )


def rprint(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
    """
    Custom print function that adds a timestamp, the pyabsa version, and a newline character before and after the printed message.

    Args:
        *objects: Any number of objects to be printed
        sep (str, optional): Separator between objects. Defaults to " ".
        end (str, optional): Ending character after all objects are printed. Defaults to "\n".
        file (io.TextIOWrapper, optional): Text file to write printed output to. Defaults to sys.stdout.
        flush (bool, optional): Whether to flush output buffer after printing. Defaults to False.
    """
    print(
        time.strftime(
            "\n[%Y-%m-%d %H:%M:%S] ({})\n".format(pyabsa_version),
            time.localtime(time.time()),
        ),
        *objects,
        sep=sep,
        end=end,
        file=file,
        flush=flush
    )


def init_optimizer(optimizer):
    """
    Initialize the optimizer for the PyTorch model.

    Args:
        optimizer: str or PyTorch optimizer object.

    Returns:
        PyTorch optimizer object.

    Raises:
        KeyError: If the optimizer is unsupported.
    """
    optimizers = {
        "adadelta": torch.optim.Adadelta,  # default lr=1.0
        "adagrad": torch.optim.Adagrad,  # default lr=0.01
        "adam": torch.optim.Adam,  # default lr=0.001
        "adamax": torch.optim.Adamax,  # default lr=0.002
        "asgd": torch.optim.ASGD,  # default lr=0.01
        "rmsprop": torch.optim.RMSprop,  # default lr=0.01
        "sgd": torch.optim.SGD,
        "adamw": torch.optim.AdamW,
        torch.optim.Adadelta: torch.optim.Adadelta,  # default lr=1.0
        torch.optim.Adagrad: torch.optim.Adagrad,  # default lr=0.01
        torch.optim.Adam: torch.optim.Adam,  # default lr=0.001
        torch.optim.Adamax: torch.optim.Adamax,  # default lr=0.002
        torch.optim.ASGD: torch.optim.ASGD,  # default lr=0.01
        torch.optim.RMSprop: torch.optim.RMSprop,  # default lr=0.01
        torch.optim.SGD: torch.optim.SGD,
        torch.optim.AdamW: torch.optim.AdamW,
    }
    if optimizer in optimizers:
        return optimizers[optimizer]
    elif hasattr(torch.optim, optimizer.__name__):
        return optimizer
    else:
        raise KeyError(
            "Unsupported optimizer: {}. "
            "Please use string or the optimizer objects in torch.optim as your optimizer".format(
                optimizer
            )
        )
