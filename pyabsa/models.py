# -*- coding: utf-8 -*-
# file: models.py
# time: 2021/6/11 0011
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os.path

from google_drive_downloader import GoogleDriveDownloader as gdd


def download_pretrained_model(task='apc', language='chinese', archive_path='', model_name='any_model'):
    print('Please check https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC '
          'to download more trained models. \n '
          'The pretrained models are used for build demo, either fine-tuned the hyper-parameters'
          ' nor trained on enough resources, it is recommended to train the models on your own custom dataset')
    tmp_dir = '{}_{}_TRAINED_MODEL'.format(task.upper(), language.upper())
    dest_path = os.path.join('.', tmp_dir)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    if len(os.listdir(dest_path)):
        return dest_path
    save_path = os.path.join(dest_path, '{}.zip'.format(model_name))
    try:
        gdd.download_file_from_google_drive(file_id=archive_path,
                                            dest_path=save_path,
                                            unzip=True)
    except:
        raise ConnectionError('Download failed, you can download the trained model manually at: {},'.format(
            'https://drive.google.com/drive/folders/1yiMTucHKy2hAx945lgzhvb9QeHvJrStC'))
    os.remove(save_path)
    return dest_path


class APCTrainedModelManger:
    ChineseModel = '1dPvXgQIQn3c2VkWjW3iE4o_A7oWfjnWv'
    EnglishModel = '1QyRM3RrnCjz293G3pol9jJM8CShAZuof'
    MultilingualModel = '1K4tCPDmvuULAmGoerIHJApWnoCAJi1p-'

    @staticmethod
    def get_Chinese_APC_trained_model():
        return download_pretrained_model(task='apc',
                                         language='chinese',
                                         archive_path=APCTrainedModelManger.ChineseModel)

    @staticmethod
    def get_English_APC_trained_model():
        return download_pretrained_model(task='apc',
                                         language='english',
                                         archive_path=APCTrainedModelManger.EnglishModel)

    @staticmethod
    def get_Multilingual_APC_trained_model():
        return download_pretrained_model(task='apc',
                                         language='multilingual',
                                         archive_path=APCTrainedModelManger.MultilingualModel)


class ATEPCTrainedModelManager:
    ChineseModel = '19VdszKYWTVL4exaSTU5zl3ueP5FNbKeJ'
    EnglishModel = '14cLWoF-yKV64D0u7Fq_k_fYbJY4hjF4L'
    MultilingualModel = '1CrAwc6Rhxrb4EDNEdCZ_cH2Pj7Q-SVkU'

    @staticmethod
    def get_English_ATEPC_trained_model():
        return download_pretrained_model(task='atepc',
                                         language='english',
                                         archive_path=ATEPCTrainedModelManager.EnglishModel)

    @staticmethod
    def get_Multilingual_ATEPC_trained_model():
        return download_pretrained_model(task='atepc',
                                         language='multilingual',
                                         archive_path=ATEPCTrainedModelManager.MultilingualModel)


class APCModelList:
    BERT_BASE = 'bert_base'
    BERT_SPC = 'bert_spc'
    LCF_BERT = 'lcf_bert'
    LCFS_BERT = 'lcfs_bert'
    SLIDE_LCF_BERT = 'slide_lcf_bert'
    SLIDE_LCFS_BERT = 'slide_lcfs_bert'
    LCA_BERT = 'lca_bert'


class ATEPCModelList:
    BERT_BASE = 'bert_base'
    LCF_ATEPC = 'lcf_atepc'
