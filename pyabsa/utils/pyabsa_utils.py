# -*- coding: utf-8 -*-
# file: pyabsa_utils.py
# time: 2021/5/20 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os


def get_auto_device():
    import torch
    gpu_name = ''
    choice = -1
    if torch.cuda.is_available():
        from pyabsa.utils.Pytorch_GPUManager import GPUManager
        gpu_name, choice = GPUManager().auto_choice()
    return gpu_name, choice


def find_target_file(dir_path, file_type, exclude_key='', find_all=False):
    '''
    'file_type': find a set of files whose name contain the 'file_type',
    'exclude_key': file name contains 'exclude_key' will be ignored
    'find_all' return a result list if Ture else the first target file
    '''

    if not find_all:
        if not dir_path:
            return ''
        elif os.path.isfile(dir_path):
            if file_type in dir_path.lower() and not (exclude_key and exclude_key in dir_path.lower()):
                return dir_path
            else:
                return ''
        elif os.path.isdir(dir_path):
            tmp_files = [p for p in os.listdir(dir_path)
                         if file_type in p.lower()
                         and not (exclude_key and exclude_key in p.lower())]
            return os.path.join(dir_path, tmp_files[0]) if tmp_files else []
        else:
            # print('No target(s) file found!')
            return ''
    else:
        if not dir_path:
            return []
        elif os.path.isfile(dir_path):
            if file_type in dir_path.lower() and not (exclude_key and exclude_key in dir_path.lower()):
                return [dir_path]
            else:
                return []
        elif os.path.isdir(dir_path):
            tmp_res = []
            tmp_files = os.listdir(dir_path)
            for file in tmp_files:
                tmp_res += find_target_file(os.path.join(dir_path, file), file_type, exclude_key, find_all)
            return tmp_res
        else:
            # print('No target file (file type:{}) found in {}!'.format(file_type, dir_path))
            return []
