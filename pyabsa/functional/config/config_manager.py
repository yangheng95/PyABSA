# -*- coding: utf-8 -*-
# file: main.py
# time: 2021/8/8
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from argparse import Namespace

import torch

import pyabsa

one_shot_messages = set()


def config_check(args):
    try:
        if 'SRD' in args:
            assert args['SRD'] >= 0
        if 'lcf' in args:
            assert args['lcf'] in {'cdw', 'cdm', 'fusion'}
        if 'window' in args:
            assert args['window'] in {'l', 'r', 'lr'}
        if 'eta' in args:
            assert args['eta'] == -1 or 0 <= args['eta'] <= 1
        if 'similarity_threshold' in args:
            assert 0 <= args['similarity_threshold'] <= 1
        if 'evaluate_begin' in args:
            assert 0 <= args['evaluate_begin'] < args['num_epoch']
        if 'cross_validate_fold' in args:
            assert args['cross_validate_fold'] == -1 or args['cross_validate_fold'] > 1
            if not 5 <= args['cross_validate_fold'] <= 10 and not args['cross_validate_fold'] == -1:
                message = 'Warning! cross_validate_fold will be better in [5, 10], instead of {}'.format(args['cross_validate_fold'])
                if message not in one_shot_messages:
                    print(message)
                    one_shot_messages.add(message)
        if 'dlcf_a' in args:
            assert args['dlcf_a'] > 1
        if 'dca_p' in args:
            assert args['dca_p'] >= 1
        if 'dca_layer' in args:
            assert args['dca_layer'] >= 1
        if args['model'] == pyabsa.APCModelList.LCA_BERT:
            assert args['lcf'] == 'cdm'  # LCA-Net only support CDM mode
        if 'ensemble_mode' in args:
            assert args['ensemble_mode'] in {'cat', 'mean'}
        if 'optimizer' in args:
            if 'radam' == args['optimizer'] or 'nadam' == args['optimizer'] or 'sparseadam' == args['optimizer'] and torch.version.__version__ < '1.10.0':
                message = 'Optimizer {} is not available in PyTorch < 1.10, it will be redirected to Adam instead.'.format(args['optimizer'])
                if message not in one_shot_messages:
                    print(message)
                    one_shot_messages.add('Optimizer {} is not available in PyTorch < 1.10, it will be redirected to Adam instead.'.format(args['optimizer']))

    except AssertionError:
        raise RuntimeError('Some parameters are not valid, please see the main example.')


class ConfigManager(Namespace):

    def __init__(self, args=None, **kwargs):
        """
        The ConfigManager is a subclass of argparse.Namespace and based on parameter dict and count the call-frequency of each parameter
        :param args: A parameter dict
        :param kwargs: Same param as Namespce
        """
        if not args:
            args = {}
        super().__init__(**kwargs)

        if isinstance(args, Namespace):
            self.args = vars(args)
            self.args_call_count = {arg: 0 for arg in vars(args)}
        else:
            self.args = args
            self.args_call_count = {arg: 0 for arg in args}

    def __getattribute__(self, arg_name):
        if arg_name == 'args' or arg_name == 'args_call_count':
            return super().__getattribute__(arg_name)
        try:
            value = super().__getattribute__('args')[arg_name]
            args_call_count = super().__getattribute__('args_call_count')
            args_call_count[arg_name] += 1
            super().__setattr__('args_call_count', args_call_count)
            return value

        except Exception as e:

            return super().__getattribute__(arg_name)

    def __setattr__(self, arg_name, value):
        if arg_name == 'args' or arg_name == 'args_call_count':
            super().__setattr__(arg_name, value)
            return
        try:
            args = super().__getattribute__('args')
            args[arg_name] = value
            super().__setattr__('args', args)
            args_call_count = super().__getattribute__('args_call_count')

            if arg_name in args_call_count:
                # args_call_count[arg_name] += 1
                super().__setattr__('args_call_count', args_call_count)

            else:
                args_call_count[arg_name] = 0
                super().__setattr__('args_call_count', args_call_count)

        except Exception as e:
            super().__setattr__(arg_name, value)

        config_check(args)
