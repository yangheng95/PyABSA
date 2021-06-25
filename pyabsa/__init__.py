# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

__version__ = '0.8.7.2'
__name__ = 'pyabsa'

from .functional import train_apc, load_sentiment_classifier
from .functional import train_atepc, load_aspect_extractor

from pyabsa.utils.pyabsa_utils import find_target_file

from pyabsa.utils.generate_inferring_set_for_apc import generate_inferrence_set_for_apc
from pyabsa.utils.convert_apc_set_to_atepc import convert_apc_set_to_atepc_set

from pyabsa.config.apc_config import apc_config_handler
from pyabsa.config.atepc_config import atepc_config_handler

from pyabsa.dataset_utils import ABSADatasets, detect_dataset

from pyabsa.model_utils import APCTrainedModelManager, ATEPCTrainedModelManager

from pyabsa.model_utils import APCModelList, ATEPCModelList

from pyabsa.model_utils import update_checkpoints

from update_checker import UpdateChecker

from termcolor import colored

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
    print('Please update via pip: {}'.format(colored('pip install -U {}'.format(__name__), 'red')))
