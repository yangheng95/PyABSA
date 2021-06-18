# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

__version__ = '0.8.1.0'
__name__ = 'pyabsa'

from .functional import train_apc, load_sentiment_classifier
from .functional import train_atepc, load_aspect_extractor

from pyabsa.utils.pyabsa_utils import find_target_file

from pyabsa.utils.generate_inferring_set_for_apc import generate_inferrence_set_for_apc
from pyabsa.utils.convert_apc_set_to_atepc import convert_apc_set_to_atepc_set

from pyabsa.config.apc_config import (get_apc_param_dict_base,
                                      get_apc_param_dict_chinese,
                                      get_apc_param_dict_english,
                                      get_apc_param_dict_multilingual)

from pyabsa.config.atepc_config import (get_atepc_param_dict_base,
                                        get_atepc_param_dict_english,
                                        get_atepc_param_dict_chinese,
                                        get_atepc_param_dict_multilingual)

from pyabsa.absa_dataset import ABSADatasets

from pyabsa.models import APCTrainedModelManger, ATEPCTrainedModelManager

from update_checker import update_check

update_check(__name__, __version__)
