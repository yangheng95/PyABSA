# -*- coding: utf-8 -*-
# file: logger.py
# time: 2021/6/2 0002

# Copyright from https://www.cnblogs.com/c-x-a/p/9072234.html

import logging
import os
import sys
import time

today = time.strftime("%Y%m%d %H%M%S", time.localtime(time.time()))


def get_logger(log_path, log_name="", log_type="training_log"):
    """
    Create a logger object with file handler and console handler.

    Args:
        log_path (str): The root directory of the log files.
        log_name (str): The name of the logger.
        log_type (str): The type of the logger.

    Returns:
        logger: A configured logger object.
    """
    if not log_path:
        log_dir = os.path.join(log_path, "logs")
    else:
        log_dir = os.path.join(".", "logs")

    full_path = os.path.join(log_dir, log_name + "_" + today)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    log_path = os.path.join(full_path, "{}.log".format(log_type))
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        # Specify logger output format.
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        # File handler.
        file_handler = logging.FileHandler(log_path, encoding="utf8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Console handler.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter
        console_handler.setLevel(logging.INFO)

        # Add handlers to logger.
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set logger output level.
        logger.setLevel(logging.INFO)

    return logger
