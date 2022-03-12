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


# To replace the class defined in https://github.com/yangheng95/PyABSA/blob/release/pyabsa/functional/dataset/dataset_manager.py#L18,
# so that the inference script works on a custom dataset.
class DatasetItem(list):
    def __init__(self, dataset_name, dataset_items=None):
        super().__init__()
        if os.path.exists(dataset_name):
            print('Construct DatasetItem from {}, assign dataset_name={}...'.format(dataset_name, os.path.basename(dataset_name)))
            # Normalizing the dataset's name (or path) to not end with a '/' or '\'
            while dataset_name and dataset_name[-1] in ['/', '\\']:
                dataset_name = dataset_name[:-1]

        # Naming the dataset with the normalized folder name only
        self.dataset_name = os.path.basename(dataset_name)

        # Creating the list of items if it does not exist
        if not dataset_items:
            dataset_items = dataset_name

        if not isinstance(dataset_items, list):
            self.append(dataset_items)
        else:
            for d in dataset_items:
                self.append(d)
        self.name = self.dataset_name


class ABSADatasetList(list):
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

    # @WeiLi9811 https://github.com/WeiLi9811
    Yelp = DatasetItem('Yelp', 'Yelp')

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
    Chinese = DatasetItem('Chinese', ['Phone', 'Camera', 'Notebook', 'Car'])
    Binary_Polarity_Chinese = DatasetItem('Chinese', ['Phone', 'Camera', 'Notebook', 'Car'])
    Triple_Polarity_Chinese = DatasetItem('Chinese', ['MOOC' 'Shampoo'])

    English = DatasetItem('English', ['Laptop14', 'Restaurant14', 'Restaurant16', 'ACL_Twitter', 'MAMS', 'Television', 'TShirt', 'Yelp'])
    SemEval = DatasetItem('SemEval', ['Laptop14', 'Restaurant14', 'Restaurant16'])  # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    Restaurant = DatasetItem('Restaurant', ['Restaurant14', 'Restaurant16'])
    Multilingual = DatasetItem('Multilingual', ['Laptop14', 'Restaurant16', 'ACL_Twitter', 'MAMS', 'Television', 'TShirt', 'Yelp', 'Phone', 'Camera', 'Notebook', 'Car'])

    def __init__(self):
        dataset_list = [
            self.Laptop14, self.Restaurant14, self.Restaurant15, self.Restaurant16,
            self.ACL_Twitter, self.MAMS, self.Television, self.TShirt,
            self.Phone, self.Car, self.Notebook, self.Camera,
            self.Binary_Polarity_Chinese, self.Triple_Polarity_Chinese,
            self.Shampoo, self.MOOC,
            self.English, self.SemEval,
            self.Restaurant, self.Multilingual
        ]
        super().__init__(dataset_list)


class ClassificationDatasetList(list):
    SST1 = DatasetItem('SST1', 'SST1')
    SST2 = DatasetItem('SST2', 'SST2')

    def __init__(self):
        dataset_list = [
            self.SST1, self.SST2
        ]
        super().__init__(dataset_list)


filter_key_words = ['.py', '.ignore', '.md', 'readme', 'log', 'result', 'zip', '.state_dict', '.model', '.png', 'acc_', 'f1_']


def detect_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = {'train': [], 'test': [], 'valid': []}
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):

            print('Loading {} dataset from: {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task, 'dataset'], exclude_key=['infer', 'test.'] + filter_key_words, disable_alert=False)
            if '.augment.ignore' in str(os.listdir(search_path)):
                print(colored('There is augmented datasets available at {}, remove .ignore to activate them.'.format(search_path), 'green'))
            elif '.augment' in str(os.listdir(search_path)):
                print(colored('Augmented datasets activated at {}'.format(search_path), 'green'))
            # Our data augmentation tool can automatically improve your dataset's performance 1-2% with additional computation budget
            # Currently, our augmentation tool © is not open-source. We encourage you to
            # share your dataset at https://github.com/yangheng95/ABSADatasets, all the copyrights belong to the owner according to the licence

            dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['.inference', 'test.'] + filter_key_words)
            dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(search_path, [d, 'valid', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(search_path, [d, 'dev', task], exclude_key=['.inference', 'train.'] + filter_key_words)
        else:
            dataset_file['train'] += find_files(d, ['train', task], exclude_key=['.inference', 'test.'] + filter_key_words)
            dataset_file['test'] += find_files(d, ['test', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(d, ['valid', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            dataset_file['valid'] += find_files(d, ['dev', task], exclude_key=['.inference', 'train.'] + filter_key_words)

    if len(dataset_file['train']) == 0:
        if os.path.isdir(dataset_path.dataset_name):
            print('No train set found from: {}, unrecognized files: {}'.format(dataset_path, ', '.join(os.listdir(dataset_path.dataset_name))))
        raise RuntimeError(
            'Fail to locate dataset: {}. If you are using your own dataset, you may need rename your dataset according to {}'.format(
                dataset_path,
                'https://github.com/yangheng95/ABSADatasets#important-rename-your-dataset-filename-before-use-it-in-pyabsa')
        )
    if len(dataset_file['test']) == 0:
        print('Warning! auto_evaluate=True, however cannot find test set using for evaluating!')

    if len(dataset_path) > 1:
        print(colored('Never mixing datasets with different sentiment labels for training & inference !', 'yellow'))

    return dataset_file


def detect_infer_dataset(dataset_path, task='apc'):
    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = []
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(ClassificationDatasetList, d):
            print('Loading {} dataset from:  {}'.format(d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task, 'dataset'], exclude_key=filter_key_words, disable_alert=False)
            dataset_file += find_files(search_path, ['.inference', d], exclude_key=['train.'] + filter_key_words)
        else:
            dataset_file += find_files(d, ['.inference', task], exclude_key=['train.'] + filter_key_words)

    if len(dataset_file) == 0:
        if os.path.isdir(dataset_path.dataset_name):
            print('No inference set found from: {}, unrecognized files: {}'.format(dataset_path, ', '.join(os.listdir(dataset_path.dataset_name))))
        raise RuntimeError(
            'Fail to locate dataset: {}. If you are using your own dataset, you may need rename your dataset according to {}'.format(
                dataset_path,
                'https://github.com/yangheng95/ABSADatasets#important-rename-your-dataset-filename-before-use-it-in-pyabsa')
        )
    if len(dataset_path) > 1:
        print(colored('Never mixing datasets with different sentiment labels for training & inference !', 'yellow'))

    return dataset_file


def download_datasets_from_github(save_path):
    if not save_path.endswith('integrated_datasets'):
        save_path = os.path.join(save_path, 'integrated_datasets')

    if find_files(save_path, 'integrated_datasets', exclude_key='.git'):
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            print('Clone ABSADatasets from https://github.com/yangheng95/ABSADatasets.git')
            git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='v1.2', depth=1)
            # git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='master', depth=1)
            try:
                shutil.move(os.path.join(tmpdir, 'datasets'), '{}'.format(save_path))
            except IOError as e:
                pass
        except Exception as e:
            try:
                print('Clone ABSADatasets from https://gitee.com/yangheng95/ABSADatasets.git')
                git.Repo.clone_from('https://gitee.com/yangheng95/ABSADatasets.git', tmpdir, branch='v1.2', depth=1)
                # git.Repo.clone_from('https://github.com/yangheng95/ABSADatasets.git', tmpdir, branch='master', depth=1)
                try:
                    shutil.move(os.path.join(tmpdir, 'datasets'), '{}'.format(save_path))
                except IOError as e:
                    pass
            except Exception as e:
                print(colored('Fail to clone ABSADatasets: {}. Please check your connection...'.format(e), 'red'))
                time.sleep(3)
                download_datasets_from_github(save_path)
