# -*- coding: utf-8 -*-
# file: dataset_manager.py
# time: 2021/6/8 0008
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import tempfile
import time

import git
import findfile

from termcolor import colored

from pyabsa import TaskCodeOption
from pyabsa.augment.apc_augment.apc_augment import __perform_apc_augmentation
from pyabsa.augment.text_agument.tc_augment import __perform_tc_augmentation
from pyabsa.utils.check_utils.dataset_version_check import check_datasets_version
from pyabsa.utils.data_utils.dataset_item import DatasetItem

filter_key_words = ['.py', '.md', 'readme', 'log', 'result', 'zip',
                    '.state_dict', '.model', '.png', 'acc_', 'f1_', '.backup', '.bak']


def detect_dataset(dataset_name_or_path, task_code: TaskCodeOption = None, load_aug=False, config=None, **kwargs):
    """
    Detect dataset from dataset_path, you need to specify the task type, which can be 'apc', 'atepc' or 'tc', etc.
    :param dataset_name_or_path: dataset name or path
    :param task_code: task type, which can be 'apc', 'atepc' or 'tc', etc.
    :param load_aug: load augmented dataset
    :param config: config
    """
    logger = config.logger if config else kwargs.get('logger', None)
    check_datasets_version(logger=logger)
    if not isinstance(dataset_name_or_path, DatasetItem):
        dataset_name_or_path = DatasetItem(dataset_name_or_path)
    dataset_file = {'train': [], 'test': [], 'valid': []}

    search_path = ''
    d = ''
    for d in dataset_name_or_path:
        if not os.path.exists(d):

            if os.path.exists('integrated_datasets'):
                logger.info('Searching dataset {} in local disk...'.format(d))
            else:
                logger.info('Searching dataset {} in https://github.com/yangheng95/ABSADatasets...'.format(d))
            download_datasets_from_github(logger=logger, **kwargs)
            search_path = findfile.find_dir(os.getcwd(), [d, task_code, 'dataset'], exclude_key=['infer', 'test.'] + filter_key_words, disable_alert=False)
            if not search_path:
                raise ValueError('Cannot find dataset: {}, you may need to remove existing integrated_datasets and try again. '
                                 'Please note that if you are using keywords to let findfile search the dataset, you need to save your dataset(s)'
                                 'in integrated_datasets/{}/{} '.format(d, 'task_name', 'dataset_name'))
            if not load_aug:
                logger.info('You can set load_aug=True in a trainer to augment your dataset (English only yet) and improve performance.'.format(search_path))
                logger.info('Please use a new folder to perform new text augment if the former augment exited unexpectedly'.format(search_path))
            # Our data augment tool can automatically improve your dataset's performance 1-2% with additional computation budget
            # The project of data augment is on github: https://github.com/yangheng95/BoostAug
            # share your dataset at https://github.com/yangheng95/ABSADatasets, all the copyrights belong to the owner according to the licence

            # For pretraining checkpoints, we use all dataset set as trainer set
            if load_aug:
                dataset_file['train'] += findfile.find_files(search_path, [d, 'train', task_code], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words)
                dataset_file['test'] += findfile.find_files(search_path, [d, 'test', task_code], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words)
                dataset_file['valid'] += findfile.find_files(search_path, [d, 'valid', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words)
                dataset_file['valid'] += findfile.find_files(search_path, [d, 'dev', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words)

                from pyabsa.utils.file_utils import convert_apc_set_to_atepc_set

                if not any(['augment' in x for x in dataset_file['train']]):
                    if task_code == TaskCodeOption.Aspect_Polarity_Classification:
                        __perform_apc_augmentation(dataset_name_or_path)
                        convert_apc_set_to_atepc_set(dataset_name_or_path)
                    elif task_code == TaskCodeOption.Text_Classification:
                        __perform_tc_augmentation(dataset_name_or_path)
                    else:
                        raise ValueError('Task {} is not supported for auto-augment'.format(task_code))
            else:
                dataset_file['train'] += findfile.find_files(search_path, [d, 'train', task_code], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['test'] += findfile.find_files(search_path, [d, 'test', task_code], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += findfile.find_files(search_path, [d, 'valid', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += findfile.find_files(search_path, [d, 'dev', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])

        else:
            print('Try to load {} dataset from local disk'.format(dataset_name_or_path))
            if load_aug:
                dataset_file['train'] += findfile.find_files(d, ['train', task_code], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words)
                dataset_file['test'] += findfile.find_files(d, ['test', task_code], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words)
                dataset_file['valid'] += findfile.find_files(d, ['valid', task_code], exclude_key=['.inference', 'train.'] + filter_key_words)
                dataset_file['valid'] += findfile.find_files(d, ['dev', task_code], exclude_key=['.inference', 'train.'] + filter_key_words)
            else:
                dataset_file['train'] += findfile.find_cwd_files([d, 'train', task_code], exclude_key=['.inference', 'test.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['test'] += findfile.find_cwd_files([d, 'test', task_code], exclude_key=['.inference', 'train.', 'valid.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += findfile.find_cwd_files([d, 'dev', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])
                dataset_file['valid'] += findfile.find_cwd_files([d, 'valid', task_code], exclude_key=['.inference', 'train.', 'test.'] + filter_key_words + ['.ignore'])

    # # if we need train a checkpoint_class using as much data as possible, we can merge train, valid and test set as trainer sets
    # dataset_file['train'] = dataset_file['train'] + dataset_file['test'] + dataset_file['valid']
    # dataset_file['test'] = []
    # dataset_file['valid'] = []

    if len(dataset_file['train']) == 0:
        if os.path.isdir(d) or os.path.isdir(search_path):
            print('No train set found from: {}, detected files: {}'.format(dataset_name_or_path, ', '.join(os.listdir(d) + os.listdir(search_path))))
        raise RuntimeError(
            'Fail to locate dataset: {}. Your dataset should be in "datasets" folder end withs ".apc" or ".atepc" or "tc". If the error persists, '
            'you may need rename your dataset according to {}'.format(dataset_name_or_path,
                                                                      'https://github.com/yangheng95/ABSADatasets#important-rename-your-dataset-filename-before-use-it-in-pyabsa')
        )
    if len(dataset_file['test']) == 0:
        logger.info('Warning! auto_evaluate=True, however cannot find test set using for evaluating!')

    if len(dataset_name_or_path) > 1:
        logger.info('Please DO NOT mix datasets with different sentiment labels for trainer & inference !')

    return dataset_file


def detect_infer_dataset(dataset_name_or_path, task_code=TaskCodeOption.Aspect_Polarity_Classification, **kwargs):
    """
    Detect the inference dataset from local disk or download from GitHub
    :param dataset_name_or_path: dataset name or path
    :param task_code: task name
    :param kwargs: other arguments
    """
    logger = kwargs.get('logger', None)
    dataset_file = []
    if isinstance(dataset_name_or_path, str) and os.path.isfile(dataset_name_or_path):
        dataset_file.append(dataset_name_or_path)
        return dataset_file

    if not isinstance(dataset_name_or_path, DatasetItem):
        dataset_name_or_path = DatasetItem(dataset_name_or_path)
    for d in dataset_name_or_path:
        if not os.path.exists(d):

            if os.path.exists('integrated_datasets'):
                if logger:
                    logger.info('Try to load {} dataset from local disk'.format(d))
                else:
                    print('Try to load {} dataset from local disk'.format(d))
            else:
                if logger:
                    logger.info('Try to download {} dataset from https://github.com/yangheng95/ABSADatasets'.format(d))
                else:
                    print('Try to download {} dataset from https://github.com/yangheng95/ABSADatasets'.format(d))
            download_datasets_from_github(logger=logger)
            search_path = findfile.find_dir(os.getcwd(), [d, task_code, 'dataset'], exclude_key=filter_key_words, disable_alert=False)
            dataset_file += findfile.find_files(search_path, ['.inference', d], exclude_key=['train.'] + filter_key_words)
        else:
            dataset_file += findfile.find_files(d, ['.inference', task_code], exclude_key=['train.'] + filter_key_words)

    if len(dataset_file) == 0:
        if os.path.isdir(dataset_name_or_path.dataset_name):
            print('No inference set found from: {}, unrecognized files: {}'.format(dataset_name_or_path, ', '.join(os.listdir(dataset_name_or_path.dataset_name))))
        raise RuntimeError(
            'Fail to locate dataset: {}. If you are using your own dataset, you may need rename your dataset according to {}'.format(
                dataset_name_or_path,
                'https://github.com/yangheng95/ABSADatasets#important-rename-your-dataset-filename-before-use-it-in-pyabsa')
        )
    if len(dataset_name_or_path) > 1:
        print(colored('Please DO NOT mix datasets with different sentiment labels for trainer & inference !', 'yellow'))

    return dataset_file


def download_datasets_from_github(**kwargs):
    """
    Download datasets from GitHub
    :param kwargs: other arguments
    """
    logger = kwargs.get('logger', None)
    save_path = os.getcwd()
    if not save_path.endswith('integrated_datasets'):
        save_path = os.path.join(save_path, 'integrated_datasets')

    if findfile.find_files(save_path, 'integrated_datasets', exclude_key='.git'):
        if kwargs.get('force_download', False):
            shutil.rmtree(save_path)
            if logger:
                logger.info('Force download datasets from https://github.com/yangheng95/ABSADatasets')
            else:
                print('Force download datasets from https://github.com/yangheng95/ABSADatasets')
        else:
            if logger:
                logger.info('Datasets already exist, skip download'.format(save_path))
            else:
                print('Datasets already exist in {}, skip download'.format(save_path))
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
