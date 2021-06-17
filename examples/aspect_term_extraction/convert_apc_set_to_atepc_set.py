# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import convert_apc_set_to_atepc_set

from pyabsa.absa_dataset import Datasets
apc_datasets = Datasets.apc_datasets
convert_apc_set_to_atepc_set(apc_datasets)
