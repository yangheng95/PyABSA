# -*- coding: utf-8 -*-
# file: atepc_utils.py
# time: 2021/5/27 0027
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import re


def split_text(text):
    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+'[a-z]"
    matches = re.findall(regex, text, re.UNICODE)
    return matches
