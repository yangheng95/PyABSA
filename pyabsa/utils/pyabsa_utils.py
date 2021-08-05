# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from termcolor import colored


def get_auto_device():
    import torch
    gpu_name = ''
    choice = -1
    if torch.cuda.is_available():
        from pyabsa.utils.Pytorch_GPUManager import GPUManager
        gpu_name, choice = GPUManager().auto_choice()
    return gpu_name, choice


def print_args(opt, logger):
    for arg in vars(opt):
        if getattr(opt, arg) is not None:
            logger.info('>>> {0}: {1}'.format(arg, getattr(opt, arg)))


def check_and_fix_labels(label_set, label_name, all_data):
    # update polarities_dim, init model behind this function!
    p_min, p_max = min(label_set), max(label_set)
    if len(sorted(list(label_set))) != len(sorted(list(range(p_max - p_min + 1)))):
        raise RuntimeError('Error! Labels are not continuous!')
    elif not sorted(list(label_set)) == sorted(list(range(p_max - p_min + 1))):
        print(colored('Warning! Invalid label detected, '
                      'the labels should be continuous and positive!', 'red'))
        print('Label-fixing triggered! (You can manually refactor the labels instead.)')
        new_label_dict = {}
        for l1, l2 in zip(sorted(list(label_set)), list(range(p_max - p_min + 1))):
            new_label_dict[l1] = l2
        for item in all_data:
            try:
                item[label_name] = new_label_dict[item[label_name]]
            except:
                item.polarity = new_label_dict[item.polarity]
        print('original labels:{}'.format(list(label_set)))
        print('new labels:{}'.format(new_label_dict))
        print(colored('Polarity label-fixing done, PLEASE DO RECORD THE NEW POLARITY LABEL MAP, '
                      'as the label inferred by model also changed!', 'green'))
