# -*- coding: utf-8 -*-
# file: checkpoint_manager.py
# time: 2021/6/11 0011
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os
import sys
import zipfile
from distutils.version import StrictVersion

import gdown
from autocuda import auto_cuda
from findfile import find_files, find_file
from termcolor import colored

from pyabsa import __version__
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.core.atepc.prediction.aspect_extractor import AspectExtractor
from pyabsa.core.tc.prediction.text_classifier import TextClassifier
from pyabsa.utils.pyabsa_utils import get_device, retry


def unzip_checkpoint(zip_path):
    try:
        print('Find zipped checkpoint: {}, unzipping...'.format(zip_path))
        sys.stdout.flush()
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(zip_path.replace('.zip', ''))
        print('Done.')
    except zipfile.BadZipfile:
        print('Unzip failed'.format(zip_path))
    return zip_path.replace('.zip', '')


class CheckpointManager:
    pass


class APCCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_sentiment_classifier(checkpoint: str = None,
                                 from_drive_url: str = '',
                                 sentiment_map: dict = None,
                                 auto_device=True,
                                 eval_batch_size=128):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        :param sentiment_map: label to text index map (deprecated and has no effect)
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :param eval_batch_size: eval batch_size in modeling
                
        :return:
        """
        if os.path.exists(checkpoint):
            checkpoint_config = find_file(checkpoint, ['.config'])
        else:
            checkpoint_config = find_file(os.getcwd(), [checkpoint, '.config'])
        if checkpoint_config:
            checkpoint = os.path.dirname(checkpoint_config)
        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = APCCheckpointManager.get_checkpoint(checkpoint, from_drive_url=from_drive_url)

        sent_classifier = SentimentClassifier(checkpoint, sentiment_map=sentiment_map, eval_batch_size=eval_batch_size)
        device, device_name = get_device(auto_device)
        sent_classifier.to(device)
        return sent_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese', from_drive_url=''):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """
        if not from_drive_url:
            aspect_sentiment_classification_checkpoint = available_checkpoints('APC')
            if checkpoint.lower() in [k.lower() for k in aspect_sentiment_classification_checkpoint.keys()]:
                print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
            else:
                print(colored(
                    'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                        checkpoint), 'red'))
                sys.exit(-1)
            return download_checkpoint(task='apc',
                                       language=checkpoint.lower(),
                                       archive_path=aspect_sentiment_classification_checkpoint[checkpoint.lower()]['id'])
        elif find_file(os.getcwd(), [checkpoint, '.config']):
            return os.path.dirname(find_file(os.getcwd(), [checkpoint, '.config']))
        else:
            return download_checkpoint_from_drive_url(task='apc',
                                                      language=checkpoint.lower(),
                                                      archive_path=from_drive_url)


class ATEPCCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_aspect_extractor(checkpoint: str = None,
                             from_drive_url: str = '',
                             sentiment_map: dict = None,
                             auto_device=True,
                             eval_batch_size=128):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :param sentiment_map: label to text index map (deprecated and has no effect)
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :param eval_batch_size: eval batch_size in modeling
        
        :return:
        """
        if os.path.exists(checkpoint):
            checkpoint_config = find_file(checkpoint, ['.config'])
        else:
            checkpoint_config = find_file(os.getcwd(), [checkpoint, '.config'])
        if checkpoint_config:
            checkpoint = os.path.dirname(checkpoint_config)
        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = ATEPCCheckpointManager.get_checkpoint(checkpoint, from_drive_url=from_drive_url)

        aspect_extractor = AspectExtractor(checkpoint, sentiment_map=sentiment_map, eval_batch_size=eval_batch_size)
        device, device_name = get_device(auto_device)
        aspect_extractor.to(device)
        return aspect_extractor

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese', from_drive_url=''):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """
        if not from_drive_url:
            atepc_checkpoint = available_checkpoints('ATEPC')
            if checkpoint.lower() in [k.lower() for k in atepc_checkpoint.keys()]:
                print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
            else:
                print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
                sys.exit(-1)
            return download_checkpoint(task='atepc',
                                       language=checkpoint.lower(),
                                       archive_path=atepc_checkpoint[checkpoint]['id'])
        elif find_file(os.getcwd(), [checkpoint, '.config']):
            return os.path.dirname(find_file(os.getcwd(), [checkpoint, '.config']))
        else:

            return download_checkpoint(task='atepc',
                                       language=checkpoint.lower(),
                                       archive_path=from_drive_url)


class TextClassifierCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_text_classifier(checkpoint: str = None,
                            from_drive_url: str = '',
                            auto_device=True,
                            eval_batch_size=128):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :param eval_batch_size: eval batch_size in modeling
                
        :return:
        """
        if os.path.exists(checkpoint):
            checkpoint_config = find_file(checkpoint, ['.config'])
        else:
            checkpoint_config = find_file(os.getcwd(), [checkpoint, '.config'])
        if checkpoint_config:
            checkpoint = os.path.dirname(checkpoint_config)
        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = TextClassifierCheckpointManager.get_checkpoint(checkpoint, from_drive_url=from_drive_url)

        text_classifier = TextClassifier(checkpoint, eval_batch_size=eval_batch_size)
        device, device_name = get_device(auto_device)
        text_classifier.to(device)
        return text_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese', from_drive_url=''):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param from_drive_url: for loading shared checkpoint on google drive from a direct url, this param disable the 'checkpoint' param.
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """
        if not from_drive_url:
            text_classification_checkpoint = available_checkpoints('TC')
            if checkpoint.lower() in [k.lower() for k in text_classification_checkpoint.keys()]:
                print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
            else:
                print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
                sys.exit(-1)
            return download_checkpoint(task='TC',
                                       language=checkpoint.lower(),
                                       archive_path=text_classification_checkpoint[checkpoint.lower()]['id'])
        elif find_file(os.getcwd(), [checkpoint, '.config']):
            return os.path.dirname(find_file(os.getcwd(), [checkpoint, '.config']))
        else:
            return download_checkpoint(task='TC',
                                       language=checkpoint.lower(),
                                       archive_path=from_drive_url)


def parse_checkpoint_info(t_checkpoint_map, task='APC'):
    print('*' * 10, colored('Available {} model checkpoints for Version:{} (this version)'.format(task, __version__), 'green'), '*' * 10)
    for i, checkpoint in enumerate(t_checkpoint_map):
        checkpoint = t_checkpoint_map[checkpoint]
        try:
            c_version = checkpoint['Available Version']
        except:
            continue

        if '-' in c_version:
            min_ver, _, max_ver = c_version.partition('-')
        elif '+' in c_version:
            min_ver, _, max_ver = c_version.partition('-')
        else:
            min_ver = c_version
            max_ver = ''
        max_ver = max_ver if max_ver else 'N.A.'
        StrictVersion()
        if max_ver == 'N.A.' or StrictVersion(min_ver) <= StrictVersion(__version__) <= StrictVersion(max_ver):

            print('-' * 100)
            for key in checkpoint:
                print('{}: {}'.format(key, checkpoint[key]))
            print('-' * 100)
    return t_checkpoint_map


def available_checkpoints(task='', from_local=False):
    """
    param from_local: to load checkpoint from a customized checkpoint map,
           which means you can load checkpoint from your Google drive (if you share your model on your Google drive)
    """
    try:
        if not from_local:
            # checkpoint_url = '1jjaAQM6F9s_IEXNpaY-bQF9EOrhq0PBD'  # V1
            checkpoint_url = '1CBVGPA3xdQqdkFFwzO5T2Q4reFtzFIJZ'  # V2
            # checkpoint_url = 'https://drive.google.com/file/d/1CBVGPA3xdQqdkFFwzO5T2Q4reFtzFIJZ/'
            if os.path.isfile('./checkpoints.json'):
                os.remove('./checkpoints.json')
            gdown.download(id=checkpoint_url, use_cookies=False, output='./checkpoints.json', quiet=False)
            # gdd.download_file_from_google_drive(file_id=checkpoint_url, dest_path='./checkpoints.json')
        checkpoint_map = json.load(open('./checkpoints.json', 'r'))

        t_checkpoint_map = {}
        if task:
            t_checkpoint_map = dict(checkpoint_map)[task.upper()] if task.upper() in checkpoint_map else {}
            parse_checkpoint_info(t_checkpoint_map, task)
        else:
            for task_map in checkpoint_map:
                parse_checkpoint_info(checkpoint_map[task_map], task_map)

        print(colored('There may be some checkpoints available for early versions of PyABSA, see ./checkpoints.json'.format(task, __version__), 'yellow'))

        # os.remove('./checkpoints.json')
        return t_checkpoint_map if task else checkpoint_map

    except Exception as e:
        if not from_local:
            print('Fail to query checkpoints from Google Drive, try to download checkpoint parsed from ./checkpoints.json')
            return available_checkpoints(task, True)
        else:
            print('\nFailed to query checkpoints (Error: {}), you can try manually download the checkpoints from: \n'.format(e) +
                  '[1]\tGoogle Drive\t: https://drive.google.com/file/d/1CBVGPA3xdQqdkFFwzO5T2Q4reFtzFIJZ/view?usp=sharing\n'
                  '[2]\tBaidu NetDisk\t: https://pan.baidu.com/s/1K8aYQ4EIrPm1GjQv_mnxEg (Access Code: absa)\n')
            sys.exit(-1)


def download_checkpoint(task='apc', language='chinese', archive_path='', model_name='any_model'):
    print(colored('Notice: The pretrained model are used for testing, '
                  'neither trained using fine-tuned hyper-parameters nor trained with enough steps, '
                  'it is recommended to train the model on your own custom datasets', 'red')
          )
    # if not os.path.exists('./checkpoints'):
    #     os.mkdir('./checkpoints')
    tmp_dir = '{}_{}_CHECKPOINT'.format(task.upper(), language.upper())
    dest_path = os.path.join('./checkpoints', tmp_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if (find_files(dest_path, '.model') or find_files(dest_path, '.state_dict')) and find_files(dest_path, '.config'):
        print('Checkpoint already downloaded, skip...')
        return dest_path

    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        if '/' in archive_path:
            archive_path = archive_path.split('/')[-2]

        gdown.download(id=archive_path, output=save_path)
        # gdd.download_file_from_google_drive(file_id=archive_path,
        #                                     dest_path=save_path,
        #                                     unzip=True,
        #                                     showsize=True)
    except ConnectionError as e:
        raise ConnectionError("Fail to download checkpoint: {}".format(e))
    unzip_checkpoint(save_path)
    os.remove(save_path)
    print(colored('Google Drive applies a restriction on public large file downloading,'
                  ' if you find the checkpoint downloaded is None or small, please download it via browser: {} '.format(archive_path), 'yellow'))
    return dest_path


def download_checkpoint_from_drive_url(task='apc', language='unknown_lang', archive_path='', model_name='any_model'):
    print(colored('Notice: The pretrained model are used for testing, '
                  'neither trained using fine-tuned the hyper-parameters nor trained with enough steps, '
                  'it is recommended to train the model on your own custom datasets', 'red')
          )
    # if not os.path.exists('./checkpoints'):
    #     os.mkdir('./checkpoints')
    tmp_dir = '{}_{}_CHECKPOINT'.format(task.upper(), language.upper())
    dest_path = os.path.join('./checkpoints', tmp_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if (find_files(dest_path, '.model') or find_files(dest_path, '.state_dict')) and find_files(dest_path, '.config'):
        print('Checkpoint already downloaded, skip...')
        return dest_path

    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        if '/' in archive_path:
            archive_path = archive_path.split('/')[-2]
        gdown.download(id=archive_path, output=save_path)
        # gdd.download_file_from_google_drive(file_id=archive_path,
        #                                     dest_path=save_path,
        #                                     unzip=True,
        #                                     showsize=True)
    except ConnectionError as e:
        raise ConnectionError("Fail to download checkpoint: {}".format(e))
    unzip_checkpoint(save_path)
    os.remove(save_path)
    print(colored('Google Drive applies a restriction on public large file downloading,'
                  ' if you find the checkpoint downloaded is None or small, please download it via browser: {} '.format(archive_path), 'yellow'))
    return dest_path


def load_sentiment_classifier(checkpoint: str = None,
                              sentiment_map: dict = None,
                              auto_device: bool = True):
    infer_model = SentimentClassifier(checkpoint, sentiment_map=sentiment_map)
    infer_model.to(auto_cuda()) if auto_device else infer_model.cpu()
    return infer_model
