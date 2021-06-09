# -*- coding: utf-8 -*-
# file: train_apc_using_multiple_datasets.py
# time: 2021/6/4 0004
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                  train and evaluate on your own atepc_datasets (need train and test atepc_datasets)                  #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, get_apc_param_dict_english

from pyabsa.dataset import semeval

# You can place multiple atepc_datasets file in one dir to easily train using some atepc_datasets

# for example, training_tutorials on the SemEval atepc_datasets, you can organize the dir as follow

# ATEPC同样支持多数据集集成训练，但请不要将极性标签（种类，长度）不同的数据集融合训练！
# --atepc_datasets
# ----laptop14
# ----restaurant14
# ----restaurant15
# ----restaurant16

# or
# --atepc_datasets
# ----SemEval2014
# ------laptop14
# ------restaurant14
# ----SemEval2015
# ------restaurant15
# ----SemEval2016
# ------restaurant16


save_path = 'state_dict'

sent_classifier = train_apc(parameter_dict=get_apc_param_dict_english(),           # set param_dict=None to use default model
                            dataset_path=semeval,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )



