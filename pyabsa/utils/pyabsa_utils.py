# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import pickle

import torch
from findfile import find_files
from termcolor import colored

SENTIMENT_PADDING = -999


def save_args(config, save_path):
    f = open(os.path.join(save_path), mode='w', encoding='utf8')
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write('{}: {}\n'.format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None, mode=0):
    activated_args = []
    default_args = []
    for arg in config.args:
        if config.args_call_count[arg]:
            activated_args.append('>>> {0}: {1}  --> Active'.format(arg, config.args[arg]))
        else:
            if mode == 0:
                default_args.append('>>> {0}: {1}  --> Default'.format(arg, config.args[arg]))
            else:
                default_args.append('>>> {0}: {1}  --> Not Used'.format(arg, config.args[arg]))

    for line in activated_args:
        if logger:
            logger.info(line)
        else:
            print(colored(line, 'green'))

    for line in default_args:
        if logger:
            logger.info(line)
        else:
            print(colored(line, 'yellow'))


def check_and_fix_labels(label_set, label_name, all_data):
    # update polarities_dim, init model behind this function!
    p_min, p_max = min(label_set), max(label_set)
    if len(sorted(list(label_set))) != len(sorted(list(range(p_max - p_min + 1)))):
        raise RuntimeError('Error! Labels are not continuous!')
    elif not sorted(list(label_set)) == sorted(list(range(p_max - p_min + 1))):
        print(colored('Warning! Invalid label detected, '
                      'the labels should be continuous and positive!', 'red'))
        print('Label-fixing is triggered! (You can manually refactor the labels instead.)')
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


def load_checkpoint(trainer, from_checkpoint_path):
    if from_checkpoint_path:
        model_path = find_files(from_checkpoint_path, '.model')
        state_dict_path = find_files(from_checkpoint_path, '.state_dict')
        config_path = find_files(from_checkpoint_path, '.config')

        if from_checkpoint_path:
            if not config_path:
                raise FileNotFoundError('.config file is missing!')
            config = pickle.load(open(config_path[0], 'rb'))
            if model_path:
                if config.model != trainer.opt.model:
                    print(colored('Warning, the checkpoint was not trained using {} from param_dict'.format(trainer.opt.model.__name__)), 'red')
                trainer.model = torch.load(model_path[0])
            if state_dict_path:
                trainer.model.load_state_dict(torch.load(state_dict_path[0]))
                trainer.model.opt = trainer.opt
                trainer.model.to(trainer.opt.device)
            else:
                print('.model or .state_dict file is missing!')
        else:
            print('No checkpoint found in {}'.format(from_checkpoint_path))
        print('Checkpoint loaded!')


optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW
}
