# -*- coding: utf-8 -*-
# file: view_latest_checkpoint.py
# time: 2021/6/20
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import available_checkpoints

# checkpoint_map = available_checkpoints()
# checkpoint_map = available_checkpoints(from_local=True)
checkpoint_map = available_checkpoints(from_local=False)
