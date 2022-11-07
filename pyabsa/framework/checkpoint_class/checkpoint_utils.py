# -*- coding: utf-8 -*-
# file: checkpoint_utils.py
# time: 02/11/2022 21:39
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
import json
import os
import sys
from distutils.version import StrictVersion

import gdown
import requests
import tqdm
from findfile import find_files, find_cwd_files
from packaging import version
from termcolor import colored
from pyabsa import __version__ as current_version, PyABSAMaterialHostAddress
from pyabsa.utils.file_utils.file_utils import unzip_checkpoint


def parse_checkpoint_info(t_checkpoint_map, task='APC', show_ckpts=False):
    print('*' * 10,
          colored('Available {} model checkpoints for Version:{} (this version)'.format(task, current_version), 'green'),
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
        if max_ver == 'N.A.' or StrictVersion(min_ver) <= StrictVersion(current_version) <= StrictVersion(max_ver):

            print('-' * 100)
            print('Checkpoint Name: {}'.format(checkpoint_name))
            for key in checkpoint:
                print('{}: {}'.format(key, checkpoint[key]))
            print('-' * 100)
    return t_checkpoint_map


def available_checkpoints(task='', show_ckpts=False):
    try:  # from huggingface space
        checkpoint_url = PyABSAMaterialHostAddress + 'raw/main/checkpoints-v2.0.json'
        response = requests.get(checkpoint_url)
        with open('./checkpoints-v2.0.json', "w") as f:
            json.dump(response.json(), f)
    except Exception as e:
        print('Fail to download checkpoints info from huggingface space, try to download from local...')
    with open('./checkpoints-v2.0.json', 'r', encoding='utf8') as f:
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
        if max_ver == 'N.A.' or version.parse(min_ver) <= version.parse(current_version) <= version.parse(max_ver):
            if task:
                t_checkpoint_map.update(checkpoint_map[c_version][task.upper()] if task.upper() in checkpoint_map[c_version] else {})
                if show_ckpts:
                    parse_checkpoint_info(t_checkpoint_map, task, show_ckpts)

    return t_checkpoint_map if task else checkpoint_map


def download_checkpoint(task: str, language: str, checkpoint: dict):
    print(colored('Notice: The pretrained model are used for testing, '
                  'it is recommended to train the model on your own custom datasets', 'red')
          )
    huggingface_checkpoint_url = PyABSAMaterialHostAddress + 'resolve/main/checkpoints/{}/{}/{}'.format(
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
