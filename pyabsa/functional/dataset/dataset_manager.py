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

    # Chinese
    Phone = DatasetItem('Phone', 'Phone')
    Car = DatasetItem('Car', 'Car')
    Notebook = DatasetItem('Notebook', 'Notebook')
    Camera = DatasetItem('Camera', 'Camera')

    # brightgems@github https://github.com/brightgems
    Shampoo = DatasetItem('Shampoo', 'Shampoo')

    # jmc123@github https://github.com/jmc-123
    MOOC = DatasetItem('MOOC', 'MOOC')

    MAMS = DatasetItem('MAMS', 'MAMS')

    # @R Mukherjee et al.
    Television = DatasetItem('Television', 'Television')
    TShirt = DatasetItem('TShirt', 'TShirt')

    # assembled dataset_utils
    Chinese = DatasetItem('Chinese', ['Chinese'])
    English = DatasetItem('English', ['laptop14', 'restaurant14', 'restaurant16', 'twitter', 'MAMS', 'Television', 'TShirt'])
    SemEval = DatasetItem('SemEval', ['laptop14', 'restaurant14', 'restaurant16'])  # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    Restaurant = DatasetItem('Restaurant', ['restaurant14', 'restaurant16'])
    Multilingual = DatasetItem('Multilingual', 'Multilingual')


class ClassificationDatasetList:
    SST1 = DatasetItem('SST1', 'SST1')
    SST2 = DatasetItem('SST2', 'SST2')


def detect_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = {'train': [], 'test': []}
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):
            print('{} dataset is loading from: {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task], exclude_key=['infer', 'test.'], disable_alert=True)
            dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['infer', 'test.', '.py', '.ignore'])
            dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['infer', 'train.', '.py', '.ignore'])
        else:
            dataset_file['train'] = find_files(d, ['train', task], exclude_key=['infer', 'test.', '.py', '.ignore'])
            dataset_file['test'] = find_files(d, ['test', task], exclude_key=['infer', 'train.', '.py', '.ignore'])

    if len(dataset_file['train']) == 0:
        raise RuntimeError('{} is not an integrated dataset or not downloaded automatically,'
                           ' or it is not a path containing datasets!'.format(dataset_path))
    if len(dataset_file['test']) == 0:
        print('Warning, auto_evaluate=True, however cannot find test set using for evaluating!')

    return dataset_file


def detect_infer_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = []
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):
            print('{} dataset is loading from: {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task], disable_alert=True)
            dataset_file += find_files(search_path, ['infer', d], ['train.', '.ignore'], '.py')
        else:
            dataset_file += find_files(d, ['infer', task], ['train.', '.ignore'], '.py')

    if len(dataset_file) == 0:
        raise RuntimeError('{} is not an integrated dataset or not downloaded automatically,'
                           ' or it is not a path containing datasets!'.format(dataset_path))

    return dataset_file


def download_datasets_from_github(save_path):
    if not save_path.endswith('integrated_datasets'):
        save_path = os.path.join(save_path, 'integrated_datasets')

    if find_files(save_path, 'integrated_datasets', exclude_key='.git'):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='master', depth=1)
            try:
                shutil.move(os.path.join(tmpdir, 'datasets'), '{}'.format(save_path))
            except IOError as e:
                pass
        except Exception as e:
            print('Fail to download datasets: {}, please check your connection to GitHub, we will keep retrying...'.format(e))
            time.sleep(3)
            download_datasets_from_github(save_path)
