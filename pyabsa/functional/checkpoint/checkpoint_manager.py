# -*- coding: utf-8 -*-
# file: checkpoint_manager.py
# time: 2021/6/11 0011
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import json
import os
import sys
import zipfile
from distutils.version import StrictVersion

import gdown
import requests
import tqdm
from findfile import find_files, find_file, find_cwd_files
from termcolor import colored

from pyabsa import __version__
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.core.atepc.prediction.aspect_extractor import AspectExtractor
from pyabsa.core.tad.prediction.tad_classifier import TADTextClassifier
from pyabsa.core.tc.prediction.text_classifier import TextClassifier
from pyabsa.utils.pyabsa_utils import retry


def unzip_checkpoint(zip_path):
    try:
        print('Find zipped checkpoint: {}, unzipping...'.format(zip_path))
        sys.stdout.flush()
        if not os.path.exists(zip_path):
            os.makedirs(zip_path.replace('.zip', ''))
        z = zipfile.ZipFile(zip_path, 'r')
        z.extractall(os.path.dirname(zip_path))
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
                                 **kwargs):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
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
            checkpoint = unzip_checkpoint(
                checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = APCCheckpointManager.get_checkpoint(checkpoint)

        sent_classifier = SentimentClassifier(checkpoint, **kwargs)
        return sent_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'multilingual'):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """
        aspect_sentiment_classification_checkpoint = available_checkpoints('APC')
        if checkpoint.lower() in [k.lower() for k in aspect_sentiment_classification_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored(
                'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                    checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task='apc',
                                   language=checkpoint.lower(),
                                   checkpoint=aspect_sentiment_classification_checkpoint[checkpoint.lower()])


class ATEPCCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_aspect_extractor(checkpoint: str = None,
                             **kwargs):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """
        if os.path.exists(checkpoint):
            checkpoint_config = find_file(checkpoint, ['.config'])
        else:
            checkpoint_config = find_file(os.getcwd(), [checkpoint, '.config'])
        if checkpoint_config:
            checkpoint = os.path.dirname(checkpoint_config)
        elif checkpoint.endswith('.zip'):
            checkpoint = unzip_checkpoint(
                checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = ATEPCCheckpointManager.get_checkpoint(checkpoint)

        aspect_extractor = AspectExtractor(checkpoint, **kwargs)
        return aspect_extractor

    @staticmethod
    def get_checkpoint(checkpoint: str = 'multilingual'):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """

        atepc_checkpoint = available_checkpoints('ATEPC')
        if checkpoint.lower() in [k.lower() for k in atepc_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task='atepc',
                                   language=checkpoint.lower(),
                                   checkpoint=atepc_checkpoint[checkpoint])


class TCCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_text_classifier(checkpoint: str = None,
                            **kwargs):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
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
            checkpoint = unzip_checkpoint(
                checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = TCCheckpointManager.get_checkpoint(checkpoint)

        text_classifier = TextClassifier(checkpoint, **kwargs)
        return text_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'multilingual'):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """

        text_classification_checkpoint = available_checkpoints('TC')
        if checkpoint.lower() in [k.lower() for k in text_classification_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task='TC',
                                   language=checkpoint.lower(),
                                   checkpoint=text_classification_checkpoint[checkpoint.lower()])


class TADCheckpointManager(CheckpointManager):
    @staticmethod
    @retry
    def get_tad_text_classifier(checkpoint: str = None,
                                eval_batch_size=128,
                                **kwargs):
        """

        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
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
            checkpoint = unzip_checkpoint(
                checkpoint if os.path.exists(checkpoint) else find_file(os.getcwd(), checkpoint))
        else:
            checkpoint = TADCheckpointManager.get_checkpoint(checkpoint)

        tad_text_classifier = TADTextClassifier(checkpoint, eval_batch_size=eval_batch_size, **kwargs)
        return tad_text_classifier

    @staticmethod
    def get_checkpoint(checkpoint: str = 'multilingual'):
        """
        download the checkpoint and return the path of the downloaded checkpoint
        :param checkpoint: zipped checkpoint name, or checkpoint path or checkpoint name queried from Google Drive
        This param is for someone wants to load a checkpoint not registered in PyABSA
        :return:
        """

        tad_classification_checkpoint = available_checkpoints('TAD')
        if checkpoint.lower() in [k.lower() for k in tad_classification_checkpoint.keys()]:
            print(colored('Downloading checkpoint:{} ...'.format(checkpoint), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint), 'red'))
            sys.exit(-1)
        return download_checkpoint(task='TAD',
                                   language=checkpoint.lower(),
                                   checkpoint=tad_classification_checkpoint[checkpoint.lower()])


def parse_checkpoint_info(t_checkpoint_map, task='APC', show_ckpts=False):
    print('*' * 10,
          colored('Available {} model checkpoints for Version:{} (this version)'.format(task, __version__), 'green'),
          '*' * 10)
    for i, checkpoint_name in enumerate(t_checkpoint_map):
        checkpoint = t_checkpoint_map[checkpoint_name]
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
        if max_ver == 'N.A.' or StrictVersion(min_ver) <= StrictVersion(__version__) <= StrictVersion(max_ver):

            print('-' * 100)
            print('Checkpoint Name: {}'.format(checkpoint_name))
            for key in checkpoint:
                print('{}: {}'.format(key, checkpoint[key]))
            print('-' * 100)
    return t_checkpoint_map


def available_checkpoints(task='', show_ckpts=False):
    try:

        try:  # from huggingface space
            checkpoint_url = 'https://huggingface.co/spaces/yangheng/Multilingual-Aspect-Based-Sentiment-Analysis/raw/main/checkpoints-v1.16.json'
            response = requests.get(checkpoint_url)
            with open('./checkpoints-v1.16.json', "w") as f:
                json.dump(response.json(), f)
        except Exception as e:
            try:  # from Google Drive
                checkpoint_url = '1CBVGPA3xdQqdkFFwzO5T2Q4reFtzFIJZ'  # V2
                gdown.download(id=checkpoint_url, use_cookies=False, output='./checkpoints-v1.16.json', quiet=False)
            except Exception as e:
                raise e
        with open('./checkpoints-v1.16.json', 'r', encoding='utf8') as f:
            checkpoint_map = json.load(f)

        t_checkpoint_map = {}
        for c_version in checkpoint_map:
            if '-' in c_version:
                min_ver, _, max_ver = c_version.partition('-')
            elif '+' in c_version:
                min_ver, _, max_ver = c_version.partition('+')
            else:
                min_ver = c_version
                max_ver = ''
            max_ver = max_ver if max_ver else 'N.A.'
            if max_ver == 'N.A.' or StrictVersion(min_ver) <= StrictVersion(__version__) <= StrictVersion(max_ver):
                if task:
                    t_checkpoint_map.update(checkpoint_map[c_version][task.upper()] if task.upper() in checkpoint_map[c_version] else {})
                    if show_ckpts:
                        parse_checkpoint_info(t_checkpoint_map, task, show_ckpts)

        print(colored(
            'There may be some checkpoints available for early versions of PyABSA, see {}'.format(task, __version__,
                                                                                                  checkpoint_url),
            'yellow'))

        # os.remove('./checkpoints.json')
        return t_checkpoint_map if task else checkpoint_map

    except Exception as e:
        print(
            '\nFailed to query checkpoints (Error: {}), you can try manually download the checkpoints from: \n'.format(
                e) +
            '[1]\tHuggingface Space (Newer)\t: https://huggingface.co/spaces/yangheng/PyABSA-ATEPC/tree/main/checkpoint\n'
            '[2]\tGoogle Drive\t: https://drive.google.com/file/d/1CBVGPA3xdQqdkFFwzO5T2Q4reFtzFIJZ/view?usp=sharing\n'
            '[2]\tBaidu NetDisk\t: https://pan.baidu.com/s/1dvGqmnGG2T7MYm0VC9jWTg (Access Code: absa)\n')
        sys.exit(-1)


def download_checkpoint(task: str, language: str, checkpoint: dict):
    print(colored('Notice: The pretrained model are used for testing, '
                  'it is recommended to train the model on your own custom datasets', 'red')
          )
    huggingface_checkpoint_url = 'https://huggingface.co/spaces/yangheng/PyABSA-ATEPC/resolve/main/checkpoint/{}/{}/{}'.format(
        checkpoint['Language'], task.upper(), checkpoint['Checkpoint File']
    )

    tmp_dir = '{}_{}_CHECKPOINT'.format(task.upper(), language.upper())
    dest_path = os.path.join('./checkpoints', tmp_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    if (find_files(dest_path, '.model') or find_files(dest_path, '.state_dict')) and find_files(dest_path, '.config'):
        print('Checkpoint already downloaded, skip...')
        return dest_path

    if find_cwd_files([checkpoint['Training Model'], checkpoint['Checkpoint File'].strip('.zip'), '.config']):
        return
    save_path = os.path.join(dest_path, checkpoint['Checkpoint File'])

    try:  # from Huggingface Space

        response = requests.get(huggingface_checkpoint_url, stream=True)

        with open(save_path, "wb") as f:
            for chunk in tqdm.tqdm(response.iter_content(chunk_size=1024 * 1024),
                                   unit='MB',
                                   total=int(response.headers['content-length']) // 1024 // 1024,
                                   postfix='Downloading checkpoint...'):
                f.write(chunk)
    except Exception as e:
        try:  # from Google Drive
            gdown.download(id=checkpoint['id'], output=save_path)
        except ConnectionError as e:
            raise ConnectionError("Fail to download checkpoint: {}".format(e))
    unzip_checkpoint(save_path)
    os.remove(save_path)
    print(colored(
        'If the auto-downloading failed, please download it via browser: {} '.format(huggingface_checkpoint_url),
        'yellow'))
    return dest_path
