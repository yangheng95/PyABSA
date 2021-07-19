# -*- coding: utf-8 -*-
# file: dataset_utils.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import git
import shutil
import tempfile

from pyabsa.utils.pyabsa_utils import find_target_file


class ABSADatasets:
    # SemEval
    Laptop14 = 'Laptop14'
    Restaurant14 = 'Restaurant14'
    Restaurant15 = 'Restaurant15'
    Restaurant16 = 'Restaurant16'

    # Twitter
    ACL_Twitter = 'Twitter'

    # Chinese
    Phone = 'Phone'
    Car = 'Car'
    Notebook = 'Notebook'
    Camera = 'Camera'
    MAMS = 'MAMS'

    # @R Mukherjee et al.
    Television = 'Television'
    TShirt = 'TShirt'

    # assembled dataset
    Chinese = 'Chinese'
    SemEval = 'SemEval'
    Multilingual = 'Multilingual'

    APC_Datasets = 'APC_Datasets'
    ATEPC_Datasets = 'ATEPC_Datasets'


def detect_dataset(dataset_path, auto_evaluate=True, task='apc_benchmark'):
    if hasattr(ABSADatasets, dataset_path) or not os.path.exists(dataset_path):
        if hasattr(ABSADatasets, dataset_path):
            print('{} is the integrated dataset, try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        else:
            print('Invalid dataset path {}, try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        dataset_name = dataset_path
        download_datasets_from_github()
        if task == 'apc_benchmark':
            dataset_path = os.path.join('datasets', 'apc_datasets')
        elif task == 'atepc_benchmark':
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
            print('Can not find test set using for evaluating!')
        if len(dataset_file) == 0:
            raise RuntimeError('Can not load train set or test set! '
                               'Make sure there are trainsets and (only) one testsets in the path:',
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
                               'Make sure there are trainsets and testsets in the path:',
                               dataset_path)
    return dataset_file


def detect_infer_dataset(dataset_path, task='apc_benchmark'):
    dataset_file = []
    if hasattr(ABSADatasets, dataset_path) or not os.path.exists(dataset_path):
        if hasattr(ABSADatasets, dataset_path):
            print('{} is the integrated dataset, try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        else:
            print('Invalid dataset path {}, try to load the dataset '
                  'from github: {}'.format(dataset_path, 'https://github.com/yangheng95/ABSADatasets'))
        dataset_name = dataset_path
        download_datasets_from_github()
        if task == 'apc_benchmark':
            dataset_path = os.path.join('datasets', 'apc_datasets')
        elif task == 'atepc_benchmark':
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
        print('Fail to remove the temp file {}'.format(t))
