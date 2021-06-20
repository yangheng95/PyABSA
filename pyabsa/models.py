# -*- coding: utf-8 -*-
# file: models.py
# time: 2021/6/11 0011
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from termcolor import colored
import os.path
import json

from google_drive_downloader import GoogleDriveDownloader as gdd


class APCModelList:
    from pyabsa.module.apc.models import BERT_BASE, BERT_SPC

    from pyabsa.module.apc.models import LCF_BERT, FAST_LCF_BERT, LCF_BERT_LARGE

    from pyabsa.module.apc.models import LCFS_BERT, FAST_LCFS_BERT, LCFS_BERT_LARGE

    from pyabsa.module.apc.models import SLIDE_LCF_BERT, SLIDE_LCFS_BERT

    from pyabsa.module.apc.models import LCA_BERT


class ATEPCModelList:
    from pyabsa.module.atepc.models import BERT_BASE_ATEPC

    from pyabsa.module.atepc.models import LCF_ATEPC, LCF_ATEPC_LARGE, FAST_LCF_ATEPC

    from pyabsa.module.atepc.models import LCFS_ATEPC, LCFS_ATEPC_LARGE, FAST_LCFS_ATEPC


def download_pretrained_model(task='apc', language='chinese', archive_path='', model_name='any_model'):
    print('Check https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC '
          'to download more trained ABSA models. \n' +
          colored('Notice: The pretrained models are used for testing, '
                  'neither trained using fine-tuned the hyper-parameters nor trained with enough steps, '
                  'it is recommended to train the models on your own custom dataset', 'red')
          )
    tmp_dir = '{}_{}_TRAINED_MODEL'.format(task.upper(), language.upper())
    dest_path = os.path.join('.', tmp_dir)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if len(os.listdir(dest_path)) > 1:
        return dest_path
    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        gdd.download_file_from_google_drive(file_id=archive_path,
                                            dest_path=save_path,
                                            unzip=True)
    except:
        raise RuntimeError(
            'Download failed, you can update PyABSA and download the trained model manually at: {},'.format(
                'https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC'))
    os.remove(save_path)
    return dest_path


class APCTrainedModelManager:
    checkpoint_archive = {
        'chinese': '1gSrfU-ALC4jbTWzaxe15Pv40iPO8Z1x2',
        'english': 'please wait upload',
        'multilingual': '1vOhLBOLQ5kjN99YwBIY9NPXNT1rAALSn',
    }

    @staticmethod
    def get_checkpoint(checkpoint_name: str = 'Chinese'):
        apc_checkpoint = update_checkpoints()['APC']
        if checkpoint_name.lower() in apc_checkpoint:
            print(colored('Find latest checkpoint named {} on Google Drive...'.format(checkpoint_name), 'green'))
        else:
            raise FileNotFoundError(colored('Checkpoint {} is not found.'.format(checkpoint_name), 'red'))
        return download_pretrained_model(task='apc',
                                         language=checkpoint_name.lower(),
                                         archive_path=apc_checkpoint[checkpoint_name.lower()])


class ATEPCTrainedModelManager:
    checkpoint_archive = {
        'chinese': '1BCIEkPwD6aMIrSs4fglKsqdAqi0Oz2VP',
        'english': 'please wait upload',
        'multilingual': 'please wait upload',
    }

    @staticmethod
    def get_checkpoint(checkpoint_name: str = 'Chinese'):
        atepc_checkpoint = update_checkpoints()['ATEPC']
        if checkpoint_name.lower() in atepc_checkpoint:
            print('Find latest checkpoint named {}'.format(checkpoint_name))
        else:
            raise FileNotFoundError(colored('Checkpoint {} is not found.'.format(checkpoint_name), 'red'))
        return download_pretrained_model(task='atepc',
                                         language=checkpoint_name.lower(),
                                         archive_path=atepc_checkpoint[checkpoint_name.lower()])


def update_checkpoints():
    try:
        checkpoint_url = '1jjaAQM6F9s_IEXNpaY-bQF9EOrhq0PBD'
        if os.path.isfile('./checkpoint_map.json'):
            os.remove('./checkpoint_map.json')
        gdd.download_file_from_google_drive(file_id=checkpoint_url,
                                            dest_path='./checkpoint_map.json')
        check_map = json.load(open('checkpoint_map.json', 'r'))
        APCTrainedModelManager.checkpoint_archive = check_map['APC']
        ATEPCTrainedModelManager.checkpoint_archive = check_map['ATEPC']
        print(colored('Available checkpoints:', 'green'), check_map)
        return check_map
    except ConnectionError as e:
        print('Failed to update available checkpoints!')
        return None
