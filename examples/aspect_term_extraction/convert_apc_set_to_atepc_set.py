# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.utils.file_utils import convert_apc_set_to_atepc_set
from pyabsa.dataset import ABSADatasetList

apc_datasets = ABSADatasetList.APC_Datasets

convert_apc_set_to_atepc_set(apc_datasets)  # for custom dataset_utils, absolute path recommended for this function
