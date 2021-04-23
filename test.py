# -*- coding: utf-8 -*-
# file: test.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from sentinfer.functional import train, load_trained_model

# from lcabsa.train.main import train, load_trained_model
param_dict = {'model_name': 'lcf_bert', 'batch_size': 16}
train_set_path = 'restaurant_train.raw'
model_path_to_save = './'
train(param_dict, train_set_path, model_path_to_save)
test_set_path = 'rest16_test_inferring.dat'
infermodel = load_trained_model(param_dict, model_path_to_save)
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'
infermodel.batch_infer(test_set_path)
