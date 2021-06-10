# -*- coding: utf-8 -*-
# file: dataset.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import git
import shutil
import tempfile

from pyabsa.utils.pyabsa_utils import find_target_file

laptop14 = 'laptop14'
restaurant14 = 'restaurant14'
restaurant15 = 'restaurant15'
restaurant16 = 'restaurant16'
acl_twitter = 'twitter'
phone = 'phone'
car = 'car'
notebook = 'notebook'
camera = 'camera'
mams = 'mams'
multilingual = 'multilingual'

chinese = 'chinese'
semeval = 'semeval'

apc_datasets = 'apc_datasets'
atepc_datasets = 'atepc_datasets'

integrated_dataset_list = {'laptop14', 'restaurant14', 'restaurant15', 'restaurant16',
                           'twitter', 'phone', 'notebook', 'camera', 'car', 'mams',
                           'multilingual', 'apc_dataset', 'atepc_datasets', 'chinese', 'semeval'}


def detect_dataset(dataset_path, auto_evaluate=True, task='apc'):
    if dataset_path.lower() in integrated_dataset_list or not os.path.exists(dataset_path):
        if dataset_path.lower() in integrated_dataset_list:
            print('{} is the integrated dataset,try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        else:
            print('Maybe {} is the integrated dataset,try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        dataset_name = dataset_path
        download_datasets_from_github()
        if task == 'apc':
            dataset_path = os.path.join('datasets', 'apc_datasets')
        elif task == 'atepc':
            dataset_path = os.path.join('datasets', 'atepc_datasets')
        else:
            raise RuntimeError('No dataset was found!')

        dataset_file = dict()
        dataset_file['train'] = find_target_file(dataset_path, 'train', exclude_key='infer', find_all=True)
        dataset_file['train'] = [d for d in dataset_file['train'] if dataset_name.lower() in d.lower()]
        if auto_evaluate and find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True):
            dataset_file['test'] = find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True)
            dataset_file['test'] = [d for d in dataset_file['test'] if dataset_name.lower() in d.lower()]
        if auto_evaluate and not find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True):
            print('Cna not find test set using for evaluating!')
        if len(dataset_file) == 0:
            raise RuntimeError('Can not load train set or test set! '
                               'Make sure there are (only) one trainset and (only) one testset in the path:',
                               dataset_path)

    else:
        dataset_file = dict()
        dataset_file['train'] = find_target_file(dataset_path, 'train', exclude_key='infer', find_all=True)
        if auto_evaluate and find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True):
            dataset_file['test'] = find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True)
        if auto_evaluate and not find_target_file(dataset_path, 'test', exclude_key='infer', find_all=True):
            print('Cna not find test set using for evaluating!')
        if len(dataset_file) == 0:
            raise RuntimeError('Can not load train set or test set! '
                               'Make sure there are (only) one trainset and (only) one testset in the path:',
                               dataset_path)
    return dataset_file


def detect_infer_dataset(dataset_path, task='apc'):
    dataset_file = []
    if dataset_path.lower() in integrated_dataset_list or not os.path.exists(dataset_path):
        if dataset_path.lower() in integrated_dataset_list:
            print('{} is the integrated dataset,try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        else:
            print('Maybe {} is the integrated dataset,try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        dataset_name = dataset_path
        download_datasets_from_github()
        if task == 'apc':
            dataset_path = os.path.join('datasets', 'apc_datasets')
        elif task == 'atepc':
            dataset_path = os.path.join('datasets', 'atepc_datasets')
        else:
            raise RuntimeError('No dataset was found!')
        dataset_file = find_target_file(dataset_path, 'infer', find_all=True)
        dataset_file = [d for d in dataset_file if dataset_name.lower() in d.lower()]
        if len(dataset_file) == 0:
            raise RuntimeError('Can not load train set or test set! '
                               'Make sure there are (only) one trainset and (only) one testset in the path:',
                               dataset_path)

    return dataset_file


def download_datasets_from_github(save_path='./'):
    if os.path.exists(os.path.join(save_path, 'datasets')):
        return
    # Create temporary dir
    t = tempfile.mkdtemp()
    try:
        # Clone into temporary dir
        git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', t, branch='master', depth=1)
    except Exception as e:
        raise e

    try:
        # Copy desired file from temporary dir
        shutil.move(os.path.join(t, 'datasets'), save_path)
    except Exception as e:
        print('Seems datasets downloaded in: {}, if not please remove the datasets and download again',
              os.path.join(save_path, 'datasets'))

    try:
        shutil.rmtree(t)
    except:
        print('fail to remove the temp file {}'.format(t))
