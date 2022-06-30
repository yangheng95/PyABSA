# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/21 0021
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

# This function coverts a ABSA datasets to inference set, try to convert every datasets found in the dir
# please do check the output file!
from pyabsa.functional import ABSADatasetList

from pyabsa.utils.file_utils import generate_inference_set_for_apc

generate_inference_set_for_apc('dataset')
