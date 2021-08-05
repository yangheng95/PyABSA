# -*- coding: utf-8 -*-
# file: train_text_classification_bert.py
# time: 2021/8/5
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

from pyabsa import train_text_classifier, classification_config_handler, ClassificationDatasetList
from pyabsa.model_utils import ClassificationModelList

save_path = 'state_dict'
classification_param_dict_english = classification_config_handler.get_classification_param_dict_english()
classification_param_dict_english['model'] = ClassificationModelList.BERTClassificationModelList.BERT
classification_param_dict_english['num_epoch'] = 10
classification_param_dict_english['evaluate_begin'] = 2
classification_param_dict_english['max_seq_len'] = 80
classification_param_dict_english['dropout'] = 0.5
classification_param_dict_english['seed'] = {42, 56, 1}
classification_param_dict_english['log_step'] = 5
classification_param_dict_english['l2reg'] = 0.0001

SST2 = ClassificationDatasetList.SST2
sent_classifier = train_text_classifier(parameter_dict=classification_param_dict_english,  # set param_dict=None to use default model
                                        dataset_path=SST2,  # train set and test set will be automatically detected
                                        model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                                        auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                                        auto_device=True  # automatic choose CUDA or CPU
                                        )

SST1 = ClassificationDatasetList.SST1
sent_classifier = train_text_classifier(parameter_dict=classification_param_dict_english,  # set param_dict=None to use default model
                                        dataset_path=SST1,  # train set and test set will be automatically detected
                                        model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                                        auto_evaluate=True,  # evaluate model while training_tutorials if test set is available
                                        auto_device=True  # automatic choose CUDA or CPU
                                        )