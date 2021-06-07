# -*- coding: utf-8 -*-
# file: generate_inference_set.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import convert_apc_set_to_atepc_set

# covert all dataset file to atepc datasets found in the target dir
# please do check the output file!
convert_apc_set_to_atepc_set(r'../../aspect_polarity_classification/apc_datasets')
