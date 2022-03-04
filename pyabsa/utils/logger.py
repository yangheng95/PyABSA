# -*- coding: utf-8 -*-
# file: logger.py
# time: 2021/6/2 0002

# Copyright from https://www.cnblogs.com/c-x-a/p/9072234.html

import logging
import os
import sys
import time

import termcolor

today = time.strftime('%Y%m%d %H%M%S', time.localtime(time.time()))


def get_logger(log_path, log_name='', log_type='training_log'):
    if not log_path:
        log_dir = os.path.join(log_path, "logs")
    else:
        log_dir = os.path.join('.', "logs")

    full_path = os.path.join(log_dir, log_name + '_' + today)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    log_path = os.path.join(full_path, "{}.log".format(log_type))
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        # 指定logger输出格式
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        # 文件日志
        file_handler = logging.FileHandler(log_path, encoding="utf8")
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
        file_handler.setLevel(logging.INFO)

        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter  # 也可以直接给formatter赋值
        console_handler.setLevel(logging.INFO)

        # 为logger添加的日志处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.INFO)

    return logger
