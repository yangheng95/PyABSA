# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95

# Copyright (C) 2021. All Rights Reserved.

__version__ = '0.9.2.1'
__name__ = 'pyabsa'

from termcolor import colored
from update_checker import UpdateChecker

from pyabsa.config.apc_config import apc_config_handler
from pyabsa.config.atepc_config import atepc_config_handler
from pyabsa.config.classification_config import classification_config_handler
from pyabsa.dataset_utils import ABSADatasetList, ABSADatasets, ClassificationDatasetList, detect_dataset
from pyabsa.functional import (train_apc,
                               load_sentiment_classifier,
                               train_atepc,
                               load_aspect_extractor,
                               train_text_classifier,
                               load_text_classifier)
from pyabsa.model_utils import (APCCheckpointManager,
                                ATEPCCheckpointManager,
                                APCTrainedModelManager,
                                ATEPCTrainedModelManager,
                                APCModelList,
                                ATEPCModelList,
                                ClassificationModelList,
                                update_checkpoints)
from pyabsa.utils import check_update_log

checker = UpdateChecker()
check_result = checker.check(__name__, __version__)

if check_result:
    print(check_result)
    check_update_log()
    print('You can update via pip: {}'.format(colored('pip install -U {}'.format(__name__), 'green')))
    # print(colored('The version ends with letter-postfix is a test version,'
    #               ' please always update if you are using a test version.', 'red'))
