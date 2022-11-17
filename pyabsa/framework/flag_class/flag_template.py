# -*- coding: utf-8 -*-
# file: flag_template.py
# time: 02/11/2022 17:13
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


class TaskNameOption(dict):
    code2name = {
        'apc': 'Aspect-based Sentiment Classification',
        'atepc': 'Aspect Term Extraction and Polarity Classification',
        'tc': 'Text Classification',
        'text_classification': 'Text Classification',
        'tad': 'Text Adversarial Defense',
        'rnac_datasets': 'RNA Sequence Classification',
        'rnar': 'RNA Sequence Regression',
        'APC': 'Aspect-based Sentiment Classification',
        'ATEPC': 'Aspect Term Extraction and Polarity Classification',
        'TC': 'Text Classification',
        'TAD': 'Text Adversarial Defense',
        'RNAC': 'RNA Sequence Classification',
        'RNAR': 'RNA Sequence Regression',
    }

    def __init__(self):
        super(TaskNameOption, self).__init__(self.code2name)

    def get(self, key):
        return self.code2name.get(key, 'Unknown Task')


class TaskCodeOption:
    Aspect_Polarity_Classification = 'APC'
    Aspect_Term_Extraction_and_Classification = 'ATEPC'
    Sentiment_Analysis = 'TC'
    Text_Classification = 'TC'
    Text_Adversarial_Defense = 'TAD'
    RNASequenceClassification = 'RNAC'
    RNASequenceRegression = 'RNAR'


class LabelPaddingOption:
    SENTIMENT_PADDING = -100
    LABEL_PADDING = -100


class ModelSaveOption:
    DO_NOT_SAVE_MODEL = 0
    SAVE_MODEL_STATE_DICT = 1
    SAVE_FULL_MODEL = 2
    SAVE_FINE_TUNED_PLM = 3


class ProxyAddressOption:
    CN_GITHUB_MIRROR = 'https://gitee.com/'


PyABSAMaterialHostAddress = 'https://huggingface.co/spaces/yangheng/PyABSA/'


class DeviceTypeOption:
    AUTO = True
    CPU = 'cpu'
    CUDA = 'cuda'
    ALL_CUDA = 'allcuda'
