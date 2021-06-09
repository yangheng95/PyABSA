# -*- coding: utf-8 -*-
# file: train_apc.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                  train and evaluate on your own atepc_datasets (need train and test atepc_datasets)                  #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, get_atepc_param_dict_base

from pyabsa.dataset import laptop14

save_path = 'state_dict'

apc_param_dict_base = get_atepc_param_dict_base()

# datasets_path = '../apc_datasets/SemEval/laptop14'  # automatic detect all datasets files in this path
sent_classifier = train_apc(parameter_dict=apc_param_dict_base,  # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,          # train set and test set will be automatically detected
                            model_path_to_save=save_path,        # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,                  # evaluate model while training_tutorials if test set is available
                            auto_device=True                     # automatic choose CUDA or CPU
                            )

# 如果有需要，使用以下方法自定义情感索引到情感标签的词典， 其中-999为必需的填充， e.g.,
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', -999: ''}
sent_classifier.set_sentiment_map(sentiment_map)
