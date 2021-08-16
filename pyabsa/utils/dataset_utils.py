# -*- coding: utf-8 -*-
# file: dataset_utils.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import tempfile

import git
from findfile import find_files, find_dir


class ABSADatasetList:
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

    # assembled dataset_utils
    Chinese = ['Chinese']
    SemEval = ['laptop14', 'restaurant14', 'restaurant16']  # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    Restaurant = ['restaurant14', 'restaurant16']
    Multilingual = 'Multilingual'

    APC_Datasets = 'APC_Datasets'
    ATEPC_Datasets = 'ATEPC_Datasets'


class ClassificationDatasetList:
    SST1 = 'SST1'
    SST2 = 'SST2'


def detect_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]
    dataset_file = {'train': [], 'test': []}
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d):
            print('{} dataset is not found locally, search at {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), ['dataset', task], exclude_key=['infer', 'test.'], disable_alert=True)
            dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['infer', 'test.'])
            dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['infer', 'train.'])
        else:
            dataset_file['train'] = find_files(d, ['dataset', 'train', task], exclude_key=['infer', 'test.'])
            dataset_file['test'] = find_files(d, ['dataset', 'test', task], exclude_key=['infer', 'train.'])

    if len(dataset_file['train']) == 0:
        raise RuntimeError('{} is not an integrated dataset, and it is not a path containing datasets!'.format(dataset_path))
    if len(dataset_file['test']) == 0:
        print('Waring, auto_evaluate=True, however cannot find test set using for evaluating!')

    return dataset_file


def detect_infer_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, list):
        dataset_path = [dataset_path]
    dataset_file = []
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d):
            print('{} dataset is not found locally, search at {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), ['dataset', d, task], disable_alert=True)
            dataset_file += find_files(search_path, ['infer', d], 'train.')
        else:
            dataset_file += find_files(d, ['infer', task], 'train.')

    if len(dataset_file) == 0:
        raise RuntimeError('{} is not an integrated dataset, and it is not a path containing datasets!'.format(dataset_path))

    return dataset_file


def download_datasets_from_github(save_path='./'):
    if find_dir(save_path, 'dataset', disable_alert=True):
        return
    # Create temporary dir
    t = tempfile.mkdtemp()
    try:
        # Clone into temporary dir
        git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', t, branch='master', depth=1)
    except Exception as e:
        raise print('Fail to download datasets: {}, please check your connection to GitHub and retry.'.format(e))

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
