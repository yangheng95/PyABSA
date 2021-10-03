# -*- coding: utf-8 -*-
# file: dataset_manager.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import tempfile
import time

import git
from findfile import find_files, find_dir
from termcolor import colored


class DatasetItem(list):
    def __init__(self, dataset_name, dataset_items=None):
        super().__init__()
        self.dataset_name = dataset_name

        if not dataset_items:
            dataset_items = dataset_name

        if not isinstance(dataset_items, list):
            self.append(dataset_items)
        else:
            for d in dataset_items:
                self.append(d)


class ABSADatasetList:

    # SemEval
    Laptop14 = DatasetItem('Laptop14', 'Laptop14')
    Restaurant14 = DatasetItem('Restaurant14', 'Restaurant14')
    Restaurant15 = DatasetItem('Restaurant15', 'Restaurant15')
    Restaurant16 = DatasetItem('Restaurant16', 'Restaurant16')

    # Twitter
    ACL_Twitter = DatasetItem('Twitter', 'Twitter')

    MAMS = DatasetItem('MAMS', 'MAMS')

    # @R Mukherjee et al.
    Television = DatasetItem('Television', 'Television')
    TShirt = DatasetItem('TShirt', 'TShirt')

    # Chinese (binary polarity)
    Phone = DatasetItem('Phone', 'Phone')
    Car = DatasetItem('Car', 'Car')
    Notebook = DatasetItem('Notebook', 'Notebook')
    Camera = DatasetItem('Camera', 'Camera')

    # Chinese (triple polarity)
    # brightgems@github https://github.com/brightgems
    Shampoo = DatasetItem('Shampoo', 'Shampoo')
    # jmc123@github https://github.com/jmc-123
    MOOC = DatasetItem('MOOC', 'MOOC')

    # assembled dataset_utils
    Chinese = DatasetItem('Chinese', ['Phone', 'Camera', 'Notebook', 'Car', 'Shampoo', 'MOOC'])
    English = DatasetItem('English', ['Laptop14', 'Restaurant14', 'Restaurant16', 'ACL_Twitter', 'MAMS', 'Television', 'TShirt'])
    SemEval = DatasetItem('SemEval', ['Laptop14', 'Restaurant14', 'Restaurant16'])  # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    Restaurant = DatasetItem('Restaurant', ['Restaurant14', 'Restaurant16'])
    Multilingual = DatasetItem('Multilingual', 'datasets')


class ClassificationDatasetList:
    SST1 = DatasetItem('SST1', 'SST1')
    SST2 = DatasetItem('SST2', 'SST2')


def filter_dataset(dataset_files):
    filter_key_words = ['.py', '.ignore', '.md', 'readme']
    if isinstance(dataset_files, list):
        for d in dataset_files:
            for k in filter_key_words:
                if k in d:
                    dataset_files.remove(d)
    else:
        for d in dataset_files['train']:
            for k in filter_key_words:
                if k in d:
                    dataset_files['train'].remove(d)

        for d in dataset_files['test']:
            for k in filter_key_words:
                if k in d:
                    dataset_files['test'].remove(d)

    return dataset_files


def detect_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = {'train': [], 'test': []}
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):
            print('{} dataset is loading from: {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task], exclude_key=['infer', 'test.'], disable_alert=False)
            dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['infer', 'test.'])
            dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['infer', 'train.'])
        else:
            dataset_file['train'] = find_files(d, ['train', task], exclude_key=['infer', 'test.'])
            dataset_file['test'] = find_files(d, ['test', task], exclude_key=['infer', 'train.'])

    if len(dataset_file['train']) == 0:
        raise RuntimeError('{} is not an integrated dataset or not downloaded automatically,'
                           ' or it is not a path containing datasets!'.format(dataset_path))
    if len(dataset_file['test']) == 0:
        print('Warning, auto_evaluate=True, however cannot find test set using for evaluating!')

    if len(dataset_path) > 1:
        print(colored('Never mixing datasets with different sentiment labels for training & inferring !', 'yellow'))

    return filter_dataset(dataset_file)


def detect_infer_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = []
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):
            print('{} dataset is loading from: {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task], disable_alert=False)
            dataset_file += find_files(search_path, ['infer', d], ['train.'])
        else:
            dataset_file += find_files(d, ['infer', task], ['train.'])

    if len(dataset_file) == 0:
        raise RuntimeError('{} is not an integrated dataset or not downloaded automatically,'
                           ' or it is not a path containing datasets!'.format(dataset_path))
    if len(dataset_path) > 1:
        print(colored('Never mixing datasets with different sentiment labels for training & inferring !', 'yellow'))

    return filter_dataset(dataset_file)


def download_datasets_from_github(save_path):
    if not save_path.endswith('integrated_datasets'):
        save_path = os.path.join(save_path, 'integrated_datasets')

    if find_files(save_path, 'integrated_datasets', exclude_key='.git'):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='v1.2', depth=1)
            # git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='master', depth=1)
            try:
                shutil.move(os.path.join(tmpdir, 'datasets'), '{}'.format(save_path))
            except IOError as e:
                pass
        except Exception as e:
            print('Fail to clone ABSADatasets: {}, please check your connection to GitHub, we will keep retrying...'.format(e))
            time.sleep(3)
            download_datasets_from_github(save_path)
