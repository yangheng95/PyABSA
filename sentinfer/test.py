# -*- coding: utf-8 -*-
# file: test.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from sentinfer.functional import train, load_trained_model

param_dict = {'batch_size': 32}
train_set_path = ''
model_path_to_save = ''
train(param_dict, train_set_path, model_path_to_save)
test_set_path = ''
infermodel = load_trained_model(param_dict, model_path_to_save)

infermodel.batch_infer(test_set_path)
