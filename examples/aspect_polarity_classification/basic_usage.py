# -*- coding: utf-8 -*-
# file: basic_usage.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import generate_inferring_set_for_apc

# This function coverts a ABSA dataset to inference set, try to convert every dataset found in the dir
generate_inferring_set_for_apc('datasets/restaurant14')
