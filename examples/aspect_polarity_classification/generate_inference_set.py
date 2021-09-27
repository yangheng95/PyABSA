# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

# This function coverts a ABSA dataset_utils to inference set, try to convert every dataset_utils found in the dir
# please do check the output file!
from pyabsa.functional import ABSADatasetList

from pyabsa.utils.file_utils import generate_inference_set_for_apc

# generate_inference_set_for_apc(dataset_path=ABSADatasetList.TShirt)
# generate_inference_set_for_apc(dataset_path=ABSADatasetList.Television)
generate_inference_set_for_apc(dataset_path='mooc')
