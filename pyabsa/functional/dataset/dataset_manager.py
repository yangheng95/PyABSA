# -*- coding: utf-8 -*-
# file: dataset_manager.py
# time: 2021/6/8 0008
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import sys
import tempfile
import time

import autocuda
import git
from findfile import find_files, find_dir, find_cwd_files

from pyabsa.core.apc.models import APCModelList

from pyabsa.functional.config import APCConfigManager, TCConfigManager
from termcolor import colored


# To replace the class defined in https://github.com/yangheng95/PyABSA/blob/release/pyabsa/functional/dataset/dataset_manager.py#L18,
# so that the inference script works on a custom dataset.
class DatasetItem(list):
    def __init__(self, dataset_name, dataset_items=None):
        super().__init__()
        if os.path.exists(dataset_name):
            # print('Construct DatasetItem from {}, assign dataset_name={}...'.format(dataset_name, os.path.basename(dataset_name)))
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
    Laptop14 = DatasetItem('Laptop14', '113.Laptop14')
    Restaurant14 = DatasetItem('Restaurant14', '114.Restaurant14')

    # https://github.com/zhijing-jin/ARTS_TestSet
    ARTS_Laptop14 = DatasetItem('ARTS_Laptop14', '111.ARTS_Laptop14')
    ARTS_Restaurant14 = DatasetItem('ARTS_Restaurant14', '112.ARTS_Restaurant14')

    Restaurant15 = DatasetItem('Restaurant15', '115.Restaurant15')
    Restaurant16 = DatasetItem('Restaurant16', '116.Restaurant16')

    # Twitter
    ACL_Twitter = DatasetItem('Twitter', '101.ACL_Twitter')

    MAMS = DatasetItem('MAMS', '109.MAMS')

    # @R Mukherjee et al.
    Television = DatasetItem('Television', '117.Television')
    TShirt = DatasetItem('TShirt', '118.TShirt')

    # @WeiLi9811 https://github.com/WeiLi9811
    Yelp = DatasetItem('Yelp', '119.Yelp')

    # Chinese (binary polarity)
    Phone = DatasetItem('Phone', '107.Phone')
    Car = DatasetItem('Car', '104.Car')
    Notebook = DatasetItem('Notebook', '106.Notebook')
    Camera = DatasetItem('Camera', '103.Camera')

    # Chinese (triple polarity)
    # brightgems@github https://github.com/brightgems
    # Note that the annotation strategy of this dataset is highly different from other datasets,
    # please dont mix this dataset with any other dataset in training
    Shampoo = DatasetItem('Shampoo', '108.Shampoo')
    # jmc123@github https://github.com/jmc-123
    MOOC = DatasetItem('MOOC', '105.MOOC')
    MOOC_En = DatasetItem('MOOC_En', '121.MOOC_En')

    # https://www.kaggle.com/datasets/cf7394cb629b099cf94f3c3ba87e1d37da7bfb173926206247cd651db7a8da07
    Kaggle = DatasetItem('Kaggle', '129.Kaggle')

    # assembled dataset
    Chinese = DatasetItem('Chinese', ['107.Phone', '103.Camera', '106.Notebook', '104.Car', '105.MOOC'])
    Binary_Polarity_Chinese = DatasetItem('Chinese', ['107.Phone', '103.Camera', '106.Notebook', '104.Car'])
    Triple_Polarity_Chinese = DatasetItem('Chinese3way', ['105.MOOC'])

    SemEval2016Task5 = DatasetItem('SemEval2016Task5', ['120.SemEval2016Task5'])
    Arabic_SemEval2016Task5 = DatasetItem('Arabic_SemEval2016Task5', ['122.Arabic'])
    Dutch_SemEval2016Task5 = DatasetItem('Dutch_SemEval2016Task5', ['123.Dutch'])
    Spanish_SemEval2016Task5 = DatasetItem('Spanish_SemEval2016Task5', ['127.Spanish'])
    Turkish_SemEval2016Task5 = DatasetItem('Turkish_SemEval2016Task5', ['128.Turkish'])
    Russian_SemEval2016Task5 = DatasetItem('Russian_SemEval2016Task5', ['126.Russian'])
    French_SemEval2016Task5 = DatasetItem('French_SemEval2016Task5', ['125.French'])
    English_SemEval2016Task5 = DatasetItem('English_SemEval2016Task5', ['124.English'])

    English = DatasetItem('English', ['113.Laptop14', '114.Restaurant14', '116.Restaurant16', '101.ACL_Twitter',
                                      '109.MAMS', '117.Television', '118.TShirt', '119.Yelp', '121.MOOC_En', '129.Kaggle'])

    # Abandon rest15 dataset due to data leakage, See https://github.com/yangheng95/PyABSA/issues/53
    SemEval = DatasetItem('SemEval', ['113.Laptop14', '114.Restaurant14', '116.Restaurant16'])
    Restaurant = DatasetItem('Restaurant', ['114.Restaurant14', '116.Restaurant16'])
    Multilingual = DatasetItem('Multilingual', ['113.Laptop14', '114.Restaurant14', '116.Restaurant16', '101.ACL_Twitter', '109.MAMS', '117.Television',
                                                '118.TShirt', '119.Yelp', '107.Phone', '103.Camera', '106.Notebook', '104.Car', '105.MOOC',  '129.Kaggle',
                                                '120.SemEval2016Task5', '121.MOOC_En'])

    def __init__(self):
        dataset_list = [
            self.Laptop14, self.Restaurant14, self.Restaurant15, self.Restaurant16,
            self.ACL_Twitter, self.MAMS, self.Television, self.TShirt,self.Kaggle,
            self.Phone, self.Car, self.Notebook, self.Camera, self.MOOC, self.MOOC_En,
            self.Chinese, self.Arabic_SemEval2016Task5, self.Dutch_SemEval2016Task5,
            self.Spanish_SemEval2016Task5, self.Turkish_SemEval2016Task5, self.Russian_SemEval2016Task5,
            self.French_SemEval2016Task5, self.English_SemEval2016Task5,
            self.English, self.SemEval, self.Restaurant, self.Multilingual
        ]
        super().__init__(dataset_list)


class TCDatasetList(list):
    SST1 = DatasetItem('SST5', '200.SST1')
    SST5 = DatasetItem('SST5', '200.SST1')
    SST2 = DatasetItem('SST2', '201.SST2')
    AGNews10K = DatasetItem('AGNews10K', '204.AGNews10K')
    IMDB10K = DatasetItem('IMDB10K', '202.IMDB10K')
    Yelp10K = DatasetItem('Amazon', '206.Amazon_Review_Polarity10K')
    SST = DatasetItem('SST', ['201.SST2'])

    def __init__(self):
        dataset_list = [
            self.SST5,
            self.SST2,
            self.Yelp10K,
            self.IMDB10K,
            self.AGNews10K,
        ]
        super().__init__(dataset_list)


class AdvTCDatasetList(TCDatasetList):
    pass


filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip',
                    '.state_dict', '.model', '.png', 'acc_', 'f1_', '.backup', '.bak']


def __perform_apc_augmentation(dataset, **kwargs):
    print(colored('No augmentation datasets found, performing APC augmentation. This may take a long time...', 'yellow'))
    print(colored('The augmentation tool is available at: {}'.format('https://github.com/yangheng95/BoostTextAugmentation'), 'yellow'))
    from boost_aug import ABSCBoostAug

    config = APCConfigManager.get_apc_config_english()
    config.model = APCModelList.FAST_LCF_BERT

    BoostingAugmenter = ABSCBoostAug(ROOT=os.getcwd(),
                                     CLASSIFIER_TRAINING_NUM=1,
                                     AUGMENT_NUM_PER_CASE=10,
                                     WINNER_NUM_PER_CASE=8,
                                     device=autocuda.auto_cuda())

    # auto-training after augmentation
    BoostingAugmenter.apc_boost_augment(config,  # BOOSTAUG
                                        dataset,
                                        train_after_aug=True,
                                        rewrite_cache=True,
                                        )
    sys.exit(0)


def __perform_tc_augmentation(dataset, **kwargs):
    print(colored('No augmentation datasets found, performing TC augmentation. this may take a long time...', 'yellow'))

    from boost_aug import TCBoostAug

    tc_config = TCConfigManager.get_classification_config_english()
    tc_config.log_step = -1

    BoostingAugmenter = TCBoostAug(ROOT=os.getcwd(),
                                   CLASSIFIER_TRAINING_NUM=1,
                                   WINNER_NUM_PER_CASE=8,
                                   AUGMENT_NUM_PER_CASE=16,
                                   device=autocuda.auto_cuda())

    # auto-training after augmentation
    BoostingAugmenter.tc_boost_augment(tc_config,
                                       dataset,
                                       train_after_aug=True,
                                       rewrite_cache=True,
                                       )
    sys.exit(0)


def detect_dataset(dataset_path, task='apc', load_aug=False):
    from pyabsa.utils.file_utils import validate_datasets_version
    validate_datasets_version()

    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = ''
    d = ''
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(TCDatasetList, d) or hasattr(AdvTCDatasetList, d):
            print('Dataset is not a path, treat dataset as keywords to Load {} from: {} or Search {} locally using findfile'.format(d, d, 'https://github.com/yangheng95/ABSADatasets'))
            download_datasets_from_github(os.getcwd())
            search_path = find_dir(os.getcwd(), [d, task, 'dataset'], exclude_key=['infer', 'test.'] + filter_key_words, disable_alert=False)
            if not search_path:
                raise ValueError('Cannot find dataset: {}, you may need to remove existing integrated_datasets and try again. '
                                 'Please note that if you are using keywords to let findfile search the dataset, you need to save your dataset(s)'
                                 'in integrated_datasets/{}/{} '.format(d, 'task_name', 'dataset_name'))
            if not load_aug:
                print(colored('You can set load_aug=True in a trainer to augment your dataset (English only yet) and improve performance.'.format(search_path), 'green'))
                print(colored('Please use a new folder to perform new text augmentation if the former augmentation exited unexpectedly'.format(search_path), 'green'))
            # Our data augmentation tool can automatically improve your dataset's performance 1-2% with additional computation budget
            # The project of data augmentation is on github: https://github.com/yangheng95/BoostAug
            # share your dataset at https://github.com/yangheng95/ABSADatasets, all the copyrights belong to the owner according to the licence

            # For pretraining checkpoints, we use all dataset set as training set
            if load_aug:
                dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words)
                dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words)
                dataset_file['valid'] += find_files(search_path, [d, 'valid', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words)
                dataset_file['valid'] += find_files(search_path, [d, 'dev', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words)
                from pyabsa.utils.file_utils import convert_apc_set_to_atepc_set

                if not any(['augment' in x for x in dataset_file['train']]):
                    if task == 'apc':
                        __perform_apc_augmentation(dataset_path)
                        convert_apc_set_to_atepc_set(dataset_path)
                    elif task == 'tc':
                        __perform_tc_augmentation(dataset_path)
                    else:
                        raise ValueError('Task {} is not supported for auto-augmentation'.format(task))
            else:
                dataset_file['train'] += find_files(search_path, [d, 'train', task], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['test'] += find_files(search_path, [d, 'test', task], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += find_files(search_path, [d, 'valid', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += find_files(search_path, [d, 'dev', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])

        else:
            print('Try to load {} dataset from local disk'.format(dataset_path))
            if load_aug:
                dataset_file['train'] += find_files(d, ['train', task], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words)
                dataset_file['test'] += find_files(d, ['test', task], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words)
                dataset_file['valid'] += find_files(d, ['valid', task], exclude_key=['.inference', 'train.'] + filter_key_words)
                dataset_file['valid'] += find_files(d, ['dev', task], exclude_key=['.inference', 'train.'] + filter_key_words)
            else:
                dataset_file['train'] += find_cwd_files([d, 'train', task], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['test'] += find_cwd_files([d, 'test', task], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += find_cwd_files([d, 'valid', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += find_cwd_files([d, 'valid', task], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])

    # # if we need train a checkpoint using as much data as possible, we can merge train, valid and test set as training sets
    # dataset_file['train'] = dataset_file['train'] + dataset_file['test'] + dataset_file['valid']
    # dataset_file['test'] = []
    # dataset_file['valid'] = []

    if len(dataset_file['train']) == 0:
        if os.path.isdir(d) or os.path.isdir(search_path):
            print('No train set found from: {}, detected files: {}'.format(dataset_path, ', '.join(os.listdir(d) + os.listdir(search_path))))
        raise RuntimeError(
            'Fail to locate dataset: {}. Your dataset should be in "datasets" folder end withs ".apc" or ".atepc" or "tc". If the error persists, '
            'you may need rename your dataset according to {}'.format(dataset_path,
                'https://github.com/yangheng95/ABSADatasets#important-rename-your-dataset-filename-before-use-it-in-pyabsa')
        )
    if len(dataset_file['test']) == 0:
        print('Warning! auto_evaluate=True, however cannot find test set using for evaluating!')

    if len(dataset_path) > 1:
        print(colored('Please DO NOT mix datasets with different sentiment labels for training & inference !', 'yellow'))

    return dataset_file


def detect_infer_dataset(dataset_path, task='apc'):
    dataset_file = []
    if isinstance(dataset_path, str) and os.path.isfile(dataset_path):
        dataset_file.append(dataset_path)
        return dataset_file

    if not isinstance(dataset_path, DatasetItem):
        dataset_path = DatasetItem(dataset_path)
    for d in dataset_path:
        if not os.path.exists(d) or hasattr(ABSADatasetList, d) or hasattr(TCDatasetList, d):
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
        print(colored('Please DO NOT mix datasets with different sentiment labels for training & inference !', 'yellow'))

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
