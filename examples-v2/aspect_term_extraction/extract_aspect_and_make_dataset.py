# -*- coding: utf-8 -*-
# file: make_absa_dataset.py
# time: 2022/7/7
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import make_ABSA_dataset

example = """
I am an engineer and I use matlab and stata for data analysis and currently taking Machine Learning course by Stanford which is fabulous. I did not like this course for the following reasons: 1- bad course design structure2- so confusing and inappropriate sequence
"""
with open("test.raw.txt", "w", encoding="utf-8") as f:
    f.write(example)

make_ABSA_dataset(dataset_name_or_path="test.raw.txt", checkpoint="english")
