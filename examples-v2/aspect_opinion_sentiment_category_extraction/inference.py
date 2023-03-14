# -*- coding: utf-8 -*-
# file: inference.py
# time: 23:25 2023/3/13
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2023. All Rights Reserved.

from pyabsa import ABSAInstruction

if __name__ == "__main__":
    generator = ABSAInstruction.ABSAGenerator("multilingual", device="cpu")
    example = [
        "The food is good, but the service is bad.",
        "The laptop is good, but the battery life is bad.",
    ]

    for example in example:
        generator.predict(example)
