# -*- coding: utf-8 -*-
# file: checkpoint_manager.py
# time: 2021/6/11 0011
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os.path
import sys
import zipfile

from autocuda import auto_cuda
from findfile import find_files, find_dir, find_file
from google_drive_downloader import GoogleDriveDownloader as gdd
from termcolor import colored

from pyabsa import __version__
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.core.atepc.prediction.aspect_extractor import AspectExtractor
from pyabsa.core.tc.prediction.text_classifier import TextClassifier


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


class APCCheckpointManager:
    @staticmethod
    def get_sentiment_classifier(checkpoint: str = None,
                                 sentiment_map: dict = None,
                                 auto_device=True):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param sentiment_map: label to text index map
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :return:
        """
        if find_dir(os.getcwd(), checkpoint):
            checkpoint = find_dir(os.getcwd(), checkpoint)

        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(find_file(os.getcwd(), checkpoint))

        elif not find_dir(os.getcwd(), checkpoint):
            checkpoint = APCCheckpointManager.get_checkpoint(checkpoint)

        sent_classifier = SentimentClassifier(find_dir(os.getcwd(), checkpoint), sentiment_map=sentiment_map)
        if isinstance(auto_device, str):
            device = auto_device
        elif isinstance(auto_device, bool):
            device = auto_cuda() if auto_device else 'cpu'
        else:
            device = auto_cuda()
        sent_classifier.to(device)
        return sent_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese'):
        aspect_sentiment_classification_checkpoint = available_checkpoints('APC')
        if checkpoint.lower() in [k.lower() for k in aspect_sentiment_classification_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
        else:
            print(colored(
                'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                    checkpoint), 'red'))
            sys.exit(-1)
        return download_pretrained_model(task='apc',
                                         language=checkpoint.lower(),
                                         archive_path=aspect_sentiment_classification_checkpoint[checkpoint.lower()]['id'])


class ATEPCCheckpointManager:
    @staticmethod
    def get_aspect_extractor(checkpoint: str = None,
                             sentiment_map: dict = None,
                             auto_device=True):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param sentiment_map: label to text index map
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :return:
        """
        if find_dir(os.getcwd(), checkpoint):
            checkpoint = find_dir(os.getcwd(), checkpoint)

        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(find_file(os.getcwd(), checkpoint))

        elif not find_dir(os.getcwd(), checkpoint):
            checkpoint = ATEPCCheckpointManager.get_checkpoint(checkpoint)

        aspect_extractor = AspectExtractor(find_dir(os.getcwd(), checkpoint), sentiment_map=sentiment_map)
        if isinstance(auto_device, str):
            device = auto_device
        elif isinstance(auto_device, bool):
            device = auto_cuda() if auto_device else 'cpu'
        else:
            device = auto_cuda()
        aspect_extractor.to(device)
        return aspect_extractor

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese'):
        atepc_checkpoint = available_checkpoints('ATEPC')
        if checkpoint.lower() in [k.lower() for k in atepc_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
            sys.exit(-1)
        return download_pretrained_model(task='atepc',
                                         language=checkpoint.lower(),
                                         archive_path=atepc_checkpoint[checkpoint.lower()]['id'])


class TextClassifierCheckpointManager:
    @staticmethod
    def get_text_classifier(checkpoint=None,
                            label_map=None,
                            auto_device=True):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from google drive
        :param label_map: label to text index map
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works
        :return:
        """
        if find_dir(os.getcwd(), checkpoint):
            checkpoint = find_dir(os.getcwd(), checkpoint)

        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(find_file(os.getcwd(), checkpoint))

        elif not find_dir(os.getcwd(), checkpoint):
            checkpoint = TextClassifierCheckpointManager.get_checkpoint(checkpoint)

        text_classifier = TextClassifier(find_dir(os.getcwd(), checkpoint), label_map=label_map)
        if isinstance(auto_device, str):
            device = auto_device
        elif isinstance(auto_device, bool):
            device = auto_cuda() if auto_device else text_classifier.cpu()
        else:
            device = auto_cuda()
        text_classifier.to(device)
        return text_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'Chinese'):

        text_classification_checkpoint = available_checkpoints('TextClassification')
        if checkpoint.lower() in [k.lower() for k in text_classification_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
            sys.exit(-1)
        return download_pretrained_model(task='atepc',
                                         language=checkpoint.lower(),
                                         archive_path=text_classification_checkpoint[checkpoint.lower()]['id'])


def compare_version(version1, version2):
    #  1 means greater, 0 means equal, -1 means lower
    if version1 and not version2:
        return 1
    elif version2 and not version1:
        return -1
    else:
        version1 = version1.split('.')
        version2 = version2.split('.')
        for v1, v2 in zip(version1, version2):
            if len(v1) == len(v2):
                if v1 > v2:
                    return 1
                if v2 > v1:
                    return -1
            else:
                if v1.startswith(v2):
                    return -1
                elif v2.startswith(v1):
                    return 1
                elif v1 == v2:
                    return 0
                else:
                    return int(v1 > v2)
        return 0


def parse_checkpoint_info(t_checkpoint_map, task='APC'):
    print('*' * 23, colored('Available {} model checkpoints for Version:{}'.format(task, __version__), 'green'), '*' * 23)
    for i, checkpoint in enumerate(t_checkpoint_map):
        print('-' * 100)
        print("{}. Checkpoint Name: {}\nModel: {}\nDataset: {} \nVersion: {} \nDescription:{} \nAuthor: {}".format(
            i + 1,
            checkpoint,

            t_checkpoint_map[checkpoint]['model']
            if 'model' in t_checkpoint_map[checkpoint] else '',

            t_checkpoint_map[checkpoint]['dataset']
            if 'dataset' in t_checkpoint_map[checkpoint] else '',

            t_checkpoint_map[checkpoint]['version']
            if 'version' in t_checkpoint_map[checkpoint] else '',

            t_checkpoint_map[checkpoint]['description']
            if 'description' in t_checkpoint_map[checkpoint] else '',

            t_checkpoint_map[checkpoint]['author']
            if 'author' in t_checkpoint_map[checkpoint] else ''
        ))
    print('-' * 100)
    return t_checkpoint_map


def available_checkpoints(task=''):
    try:
        checkpoint_url = '1jjaAQM6F9s_IEXNpaY-bQF9EOrhq0PBD'
        if os.path.isfile('./checkpoints.json'):
            os.remove('./checkpoints.json')
        gdd.download_file_from_google_drive(file_id=checkpoint_url, dest_path='./checkpoints.json')
        checkpoint_map = json.load(open('./checkpoints.json', 'r'))
        current_version_map = {}
        for t_map in checkpoint_map:
            if '-' in t_map:
                min_ver, _, max_ver = t_map.partition('-')
            elif '+' in t_map:
                min_ver, _, max_ver = t_map.partition('-')
            else:
                min_ver = t_map
                max_ver = ''
            max_ver = max_ver if max_ver else 'N.A.'
            if compare_version(min_ver, __version__) <= 0 and compare_version(__version__, max_ver) <= 0:
                current_version_map.update(checkpoint_map[t_map])  # add checkpoint_map[t_map]
        t_checkpoint_map = {}
        if task:
            t_checkpoint_map = dict(current_version_map)[task.upper()] if task in current_version_map else {}
            parse_checkpoint_info(t_checkpoint_map, task)
        else:
            for task_map in current_version_map:
                parse_checkpoint_info(current_version_map[task_map], task_map)

        # os.remove('./checkpoints.json')
        return t_checkpoint_map if task else current_version_map
    except Exception as e:
        print('\nFailed to query checkpoints (Error: {}), try manually download the checkpoints from: \n'.format(e) +
              '[1]\tGoogle Drive\t: https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC\n'
              '[2]\tBaidu NetDisk\t: https://pan.baidu.com/s/1K8aYQ4EIrPm1GjQv_mnxEg (Access Code: absa)\n')
        sys.exit(-1)


def download_pretrained_model(task='apc', language='chinese', archive_path='', model_name='any_model'):
    print(colored('Notice: The pretrained model are used for testing, '
                  'neither trained using fine-tuned the hyper-parameters nor trained with enough steps, '
                  'it is recommended to train the model on your own custom datasets', 'red')
          )
    # if not os.path.exists('./checkpoints'):
    #     os.mkdir('./checkpoints')
    tmp_dir = '{}_{}_TRAINED_MODEL'.format(task.upper(), language.upper())
    dest_path = os.path.join('./checkpoints', tmp_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if find_files(dest_path, '.model') and find_files(dest_path, '.config'):
        return dest_path

    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        if '/' in archive_path:
            archive_path = archive_path.split('/')[-2]
        gdd.download_file_from_google_drive(file_id=archive_path,
                                            dest_path=save_path,
                                            unzip=True,
                                            showsize=True)
    except ConnectionError as e:
        raise ConnectionError("Fail to download checkpoint: {}".format(e))
    os.remove(save_path)
    return dest_path


def load_sentiment_classifier(checkpoint: str = None,
                              sentiment_map: dict = None,
                              auto_device: bool = True):
    infer_model = SentimentClassifier(checkpoint, sentiment_map=sentiment_map)
    infer_model.to(auto_cuda()) if auto_device else infer_model.cpu()
    return infer_model
