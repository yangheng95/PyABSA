# -*- coding: utf-8 -*-
# file: functional.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
from argparse import Namespace

from pyabsa.batch_inferring.inferring import INFER_MODEL
from pyabsa.main.train import train_by_single_config
from pyabsa.main.training_configs import *


def init_training_config(config_dict):
    config = dict()

    config['SRD'] = SRD
    config['batch_size'] = batch_size
    config['distance_aware_window'] = distance_aware_window
    config['dropout'] = dropout
    config['l2reg'] = l2reg
    config['lcf'] = lcf
    config['initializer'] = initializer
    config['learning_rate'] = learning_rate
    config['max_seq_len'] = max_seq_len
    config['model_name'] = model_name
    config['num_epoch'] = num_epoch
    config['optimizer'] = optimizer
    config['pretrained_bert_name'] = pretrained_bert_name
    config['use_bert_spc'] = use_bert_spc
    config['use_dual_bert'] = use_dual_bert
    config['window'] = window
    config['seed'] = seed
    config['embed_dim'] = embed_dim
    config['hidden_dim'] = hidden_dim
    config['polarities_dim'] = polarities_dim

    if 'device' in config_dict:
        config['device'] = config_dict['device']
    else:
        try:
            from pyabsa.utils.Pytorch_GPUManager import GPUManager
            choice = GPUManager().auto_choice()
            config['device'] = 'cuda:' + str(choice)
        except:
            config['device'] = 'cpu'

    for param in config_dict:
        config[param] = config_dict[param]
    config = Namespace(**config)
    return config


def train(parameter_dict=None, train_dataset_path=None, model_path_to_save=None):
    if not train_dataset_path:
        train_dataset_path = os.getcwd()
        print('Try to load dataset in current path.')
    # load training set
    try:
        if os.path.isdir(train_dataset_path):
            train_dataset_path += '/' + [p for p in os.listdir(train_dataset_path) if 'train' in p.lower()][0]
    except:
        raise RuntimeError('Can not find path of train dataset!')
    config = init_training_config(parameter_dict)
    config.train_dataset_path = train_dataset_path
    config.model_path_to_save = model_path_to_save
    return INFER_MODEL(from_train_model=train_by_single_config(config))


def load_trained_model(trained_model_path=None):
    print('Load trained model from', trained_model_path)

    if not trained_model_path:
        trained_model_path = os.getcwd()
        print('Try to load dataset in current path.')

        raise RuntimeError('Can not find path of trained model!')
    InferModel = INFER_MODEL(trained_model_path)
    return InferModel


def print_usages():
    usages = '1. Use your data to train the model, please build a custom data set according ' \
             'to the format of the data set provided by the reference\n' \
             '利用你的数据训练模型，请根据参考提供的数据集的格式构建自定义数据集\n' \
             'infer_model = train(param_dict, train_set_path, model_path_to_save)\n' \
                \
             '2. Load the trained model\n' \
             '加载已训练并保存的模型\n' \
             'infermodel = load_trained_model(param_dict, model_path_to_save)\n' \
                \
             '3. Batch reasoning about emotional polarity based on files\n' \
             '根据文件批量推理情感极性\n' \
             'result = infermodel.batch_infer(test_set_path)\n' \
                \
             '4. Input a single text to infer sentiment\n' \
             '输入单条文本推理情感\n' \
             'infermodel.infer(text)\n' \
                \
             '5. Convert the provided dataset into a dataset for inference\n' \
             '将提供的数据集转换为推理用的数据集\n' \
             'convert_dataset_for_inferring(dataset_path)\n'

    print(usages)
