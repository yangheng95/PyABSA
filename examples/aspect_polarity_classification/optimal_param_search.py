# -*- coding: utf-8 -*-
# file: optimal_apc_param_search.py
# time: 2021/6/13 0013
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa.research.parameter_search.search_param_for_apc import apc_param_search

from pyabsa.absa_dataset import laptop14, restaurant16

from pyabsa.config.apc_config import get_apc_param_dict_english

###############################################################


# apc_param_dict_english = get_apc_param_dict_english()
# apc_param_dict_english['log_step'] = 50
# param_to_search = ['dynamic_truncate', [True, False]]
# apc_param_search(parameter_dict=apc_param_dict_english,
#                  dataset_path=laptop14,
#                  search_param=param_to_search,
#                  auto_evaluate=True,
#                  auto_device=True)

# apc_param_dict_english = get_apc_param_dict_english()
# apc_param_dict_english['log_step'] = 50
# apc_param_dict_english['evaluate_begin'] = 2
# param_to_search = ['max_seq_len', [60, 70, 80, 90, 100, 120]]
# apc_param_search(parameter_dict=apc_param_dict_english,
#                  dataset_path=laptop14,
#                  search_param=param_to_search,
#                  auto_evaluate=True,
#                  auto_device=True)

apc_param_dict_english = get_apc_param_dict_english()
apc_param_dict_english['log_step'] = 50
apc_param_dict_english['evaluate_begin'] = 2
param_to_search = ['dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]]
apc_param_search(parameter_dict=apc_param_dict_english,
                 dataset_path=restaurant16,
                 search_param=param_to_search,
                 auto_evaluate=True,
                 auto_device=True)

