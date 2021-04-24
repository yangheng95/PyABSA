# -*- coding: utf-8 -*-
# file: test.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train, convert_dataset_for_inference, get_samples, print_usages, load_trained_model

print_usages()
samples = get_samples()
convert_dataset_for_inference('datasets/semeval16')
param_dict = {'model_name': 'lcf_bert', 'batch_size': 16, 'device': 'cuda', 'num_epoch': 1}
train_set_path = 'restaurant_train.raw'
model_path_to_save = './'
infermodel = train(param_dict, train_set_path, model_path_to_save)

for sample in samples:
       infermodel.infer(sample)
test_set_path = './'
infermodel = load_trained_model(trained_model_path='lcf_bert_trained')
text = 'everything is always cooked to perfection , the [ASP]service[ASP] is excellent ,' \
       ' the [ASP]decor[ASP] cool and understated . !sent! 1 1'

infermodel.batch_infer(test_set_path, save_result=True)
