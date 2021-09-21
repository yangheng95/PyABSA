# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.utils.file_utils import convert_apc_set_to_atepc_set
from pyabsa.functional import ABSADatasetList

convert_apc_set_to_atepc_set(ABSADatasetList.Shampoo)  # for custom dataset_utils, absolute path recommended for this function
