# -*- coding: utf-8 -*-
# file: optimal_apc_param_search.py
# time: 2021/6/13 0013
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.research.parameter_search.search_param_for_apc import apc_param_search

from pyabsa.absa_dataset import Laptop14, Restaurant16

from pyabsa.config.apc_config import apc_config_handler

###############################################################


apc_param_dict_english = apc_config_handler.get_apc_param_dict_english()
apc_param_dict_english['log_step'] = 50
apc_param_dict_english['evaluate_begin'] = 2
param_to_search = ['dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
apc_param_search(parameter_dict=apc_param_dict_english,
                 dataset_path=Restaurant16,
                 search_param=param_to_search,
                 auto_evaluate=True,
                 auto_device=True)

