# -*- coding: utf-8 -*-
# file: load_result.py.py
# time: 01/05/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json

res = json.load(open('atepc_inference.result.json'))
print(res)
