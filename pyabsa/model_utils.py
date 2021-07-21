# -*- coding: utf-8 -*-
# file: model_utils.py
# time: 2021/6/11 0011
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import pyabsa.tasks.apc.models
import pyabsa.tasks.atepc.models

import pyabsa.tasks.apc.baseline.glove_apc.models

from pyabsa import __version__

from termcolor import colored
import os.path
import json

from google_drive_downloader import GoogleDriveDownloader as gdd


class APCModelList:
    SLIDE_LCFS_BERT = pyabsa.tasks.apc.models.SLIDE_LCFS_BERT
    SLIDE_LCF_BERT = pyabsa.tasks.apc.models.SLIDE_LCF_BERT

    DLCF_DCA_BERT = pyabsa.tasks.apc.models.DLCF_DCA_BERT

    LCF_BERT = pyabsa.tasks.apc.models.LCF_BERT
    FAST_LCF_BERT = pyabsa.tasks.apc.models.FAST_LCF_BERT
    LCF_DUAL_BERT = pyabsa.tasks.apc.models.LCF_DUAL_BERT

    LCFS_BERT = pyabsa.tasks.apc.models.LCFS_BERT
    FAST_LCFS_BERT = pyabsa.tasks.apc.models.FAST_LCFS_BERT
    LCFS_DUAL_BERT = pyabsa.tasks.apc.models.LCFS_DUAL_BERT

    LCA_BERT = pyabsa.tasks.apc.models.LCA_BERT

    BERT_BASE = pyabsa.tasks.apc.models.BERT_BASE
    BERT_SPC = pyabsa.tasks.apc.models.BERT_SPC

    LCF_TEMPLATE_BERT = pyabsa.tasks.apc.models.LCF_TEMPLATE_BERT

    class GloVeAPCModelList:
        LSTM = pyabsa.tasks.apc.baseline.glove_apc.models.LSTM
        IAN = pyabsa.tasks.apc.baseline.glove_apc.models.IAN
        MemNet = pyabsa.tasks.apc.baseline.glove_apc.models.MemNet
        RAM = pyabsa.tasks.apc.baseline.glove_apc.models.RAM
        TD_LSTM = pyabsa.tasks.apc.baseline.glove_apc.models.TD_LSTM
        TC_LSTM = pyabsa.tasks.apc.baseline.glove_apc.models.TC_LSTM
        Cabasc = pyabsa.tasks.apc.baseline.glove_apc.models.Cabasc
        ATAE_LSTM = pyabsa.tasks.apc.baseline.glove_apc.models.ATAE_LSTM
        TNet_LF = pyabsa.tasks.apc.baseline.glove_apc.models.TNet_LF
        AOA = pyabsa.tasks.apc.baseline.glove_apc.models.AOA
        MGAN = pyabsa.tasks.apc.baseline.glove_apc.models.MGAN
        ASGCN = pyabsa.tasks.apc.baseline.glove_apc.models.ASGCN


class ATEPCModelList:
    BERT_BASE_ATEPC = pyabsa.tasks.atepc.models.BERT_BASE_ATEPC

    LCF_ATEPC = pyabsa.tasks.atepc.models.LCF_ATEPC
    LCF_ATEPC_LARGE = pyabsa.tasks.atepc.models.LCF_ATEPC_LARGE
    FAST_LCF_ATEPC = pyabsa.tasks.atepc.models.FAST_LCF_ATEPC

    LCFS_ATEPC = pyabsa.tasks.atepc.models.LCFS_ATEPC
    LCFS_ATEPC_LARGE = pyabsa.tasks.atepc.models.LCFS_ATEPC_LARGE
    FAST_LCFS_ATEPC = pyabsa.tasks.atepc.models.FAST_LCFS_ATEPC

    LCF_TEMPLATE_ATEPC = pyabsa.tasks.atepc.models.LCF_TEMPLATE_ATEPC


def download_pretrained_model(task='apc', language='chinese', archive_path='', model_name='any_model'):
    print(colored('Notice: The pretrained models are used for testing, '
                  'neither trained using fine-tuned the hyper-parameters nor trained with enough steps, '
                  'it is recommended to train the models on your own custom dataset', 'red')
          )
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    tmp_dir = '{}_{}_TRAINED_MODEL'.format(task.upper(), language.upper())
    dest_path = os.path.join('./checkpoints', tmp_dir)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if len(os.listdir(dest_path)) > 1:
        return dest_path
    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        if '/' in archive_path:
            archive_path = archive_path.split('/')[-2]
        gdd.download_file_from_google_drive(file_id=archive_path,
                                            dest_path=save_path,
                                            unzip=True)
    except:
        raise ConnectionError("Fail to download checkpoint, seems to be a connection error.")
    os.remove(save_path)
    return dest_path


class APCTrainedModelManager:
    @staticmethod
    def get_checkpoint(checkpoint_name: str = 'Chinese'):
        apc_checkpoint = update_checkpoints('APC')
        if checkpoint_name.lower() in apc_checkpoint:
            print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint_name), 'green'))
        else:
            print(colored(
                'Checkpoint:{} is not found, you can raise an issue for requesting shares of checkpoints'.format(
                    checkpoint_name), 'red'))
            exit(-1)
        return download_pretrained_model(task='apc',
                                         language=checkpoint_name.lower(),
                                         archive_path=apc_checkpoint[checkpoint_name.lower()]['id'])


class ATEPCTrainedModelManager:

    @staticmethod
    def get_checkpoint(checkpoint_name: str = 'Chinese'):
        atepc_checkpoint = update_checkpoints('ATEPC')
        if checkpoint_name.lower() in atepc_checkpoint:
            print(colored('Downloading checkpoint:{} from Google Drive...'.format(checkpoint_name), 'green'))
        else:
            print(colored('Checkpoint:{} is not found.'.format(checkpoint_name), 'red'))
            exit(-1)
        return download_pretrained_model(task='atepc',
                                         language=checkpoint_name.lower(),
                                         archive_path=atepc_checkpoint[checkpoint_name.lower()]['id'])


def update_checkpoints(task=''):
    try:
        checkpoint_url = '1jjaAQM6F9s_IEXNpaY-bQF9EOrhq0PBD'
        if os.path.isfile('./checkpoints.json'):
            os.remove('./checkpoints.json')
        gdd.download_file_from_google_drive(file_id=checkpoint_url,
                                            dest_path='./checkpoints.json')
        checkpoint_map = json.load(open('./checkpoints.json', 'r'))
        current_version_map = {}
        for t_map in checkpoint_map:
            if '-' in t_map:
                min_ver, _, max_ver = t_map.partition('-')
                max_ver = max_ver if max_ver else 'N.A.'
                if min_ver <= __version__ <= max_ver:
                    current_version_map.update(checkpoint_map[t_map])
            elif '+' in t_map:
                min_ver, _, max_ver = t_map.partition('+')
                max_ver = max_ver if max_ver else 'N.A.'
                if min_ver <= __version__ <= max_ver:
                    current_version_map.update(checkpoint_map[t_map])
            else:
                min_ver = t_map
                if min_ver <= __version__:
                    current_version_map.update(checkpoint_map[t_map])
        APC_checkpoint_map = dict(current_version_map)['APC'] if 'APC' in current_version_map else {}
        ATEPC_checkpoint_map = dict(current_version_map)['ATEPC'] if 'ATEPC' in current_version_map else {}

        if not (task and 'APC' not in task.upper()):
            print('*' * 23, colored('Available APC model checkpoints for Version:{}'.format(__version__), 'green'),
                  '*' * 23)
            for i, checkpoint in enumerate(APC_checkpoint_map):
                print('-' * 100)
                print("{}. Checkpoint Name: {}\nDescription: {}\nComment: {}".format(
                    i + 1,
                    checkpoint,
                    APC_checkpoint_map[checkpoint]['description'],
                    APC_checkpoint_map[checkpoint]['comment'],
                ))
            print('-' * 100)
        if not (task and 'ATEPC' not in task.upper()):
            print('-' * 100)
            print('*' * 22, colored('Available ATEPC model checkpoints for Version:{}'.format(__version__), 'green'),
                  '*' * 22)
            for i, checkpoint in enumerate(ATEPC_checkpoint_map):
                print('-' * 100)
                print("{}. Checkpoint Name: {}\nDescription: {}\nComment: {}".format(
                    i + 1,
                    checkpoint,
                    ATEPC_checkpoint_map[checkpoint]['description'],
                    ATEPC_checkpoint_map[checkpoint]['comment']
                ))
            print('-' * 100)
        # os.remove('./checkpoints.json')
        return APC_checkpoint_map if task.upper() == 'APC' else ATEPC_checkpoint_map
    except Exception as e:
        print('\nFailed to query checkpoints, try manually download the checkpoints from: \n'
              '[1]\tGoogle Drive\t: https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC\n'
              '[2]\tBaidu NetDisk\t: https://pan.baidu.com/s/1K8aYQ4EIrPm1GjQv_mnxEg (Access Code: absa)\n')
        exit()
