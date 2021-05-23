# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import torch


def get_auto_device():
    choice = -1
    if torch.cuda.is_available():
        from .Pytorch_GPUManager import GPUManager
        choice = GPUManager().auto_choice()
    return choice


def find_target_file(dir_path, file_type, exclude_key='', find_all=False):
    if os.path.isfile(dir_path) and file_type in dir_path:
        return [dir_path] if find_all else dir_path
    elif os.path.isfile(dir_path) and file_type not in dir_path:
        return ''

    if not find_all:
        path = os.path.join(dir_path,
                            [p for p in os.listdir(dir_path)
                             if file_type in p.lower()
                             and not (exclude_key and exclude_key in p.lower())][0])
    else:
        path = [os.path.join(dir_path, p)
                for p in os.listdir(dir_path)
                if file_type in p.lower()
                and not (exclude_key and exclude_key in p.lower())]

    return path


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
