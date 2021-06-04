# -*- coding: utf-8 -*-
# file: train_apc_using_multiple_datasets.py
# time: 2021/6/4 0004
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_apc

# You can place multiple datasets file in one dir to easily train using some datasets

# for example, training on the SemEval datasets, you can organize the dir as fillow
# --datasets
# ----laptop14
# ----restaurant14
# ----restaurant15
# ----restaurant16

# or
# --datasets
# ----SemEval2014
# ------laptop14
# ------restaurant14
# ----SemEval2015
# ------restaurant15
# ----SemEval2016
# ------restaurant16


save_path = 'state_dict'

datasets_path = 'datasets/SemEval/restaurant15'  # file or dir are accepted
sent_classifier = train_apc(parameter_dict=None,     # set param_dict=None to use default model
                            dataset_path=datasets_path,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )



