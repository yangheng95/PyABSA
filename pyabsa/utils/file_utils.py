# -*- coding: utf-8 -*-
# file: file_utils.py
# time: 2021/7/13 0020
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import json
import os
from termcolor import colored

from google_drive_downloader import GoogleDriveDownloader as gdd

from pyabsa import __version__


def check_update_log():
    try:
        if os.path.exists('./release_note.json'):
            os.remove('./release_note.json')
        gdd.download_file_from_google_drive('1nOppewL8L1mGy9i6HQnJrEWrfaqQhC_2', './release_note.json')
        update_logs = json.load(open('release_note.json'))
        for v in update_logs:
            if v > __version__:
                print(colored('*' * 20 + ' Release Note of Version {} '.format(v) + '*' * 20, 'green'))
                for i, line in enumerate(update_logs[v]):
                    print('{}.\t{}'.format(i + 1, update_logs[v][line]))
    except:
        print('Fail to load release note of this version')
