# -*- coding: utf-8 -*-
# file: usa_configuration.py
# time: 02/11/2022 19:55
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.


import copy

from pyabsa.framework.configuration_class.configuration_template import ConfigManager
from pyabsa.tasks.UniversalSentimentAnalysis.models.model import GenerationModel

# if you find the optimal param set of some situation, e.g., some model on some datasets
# please share the main use template main
_usa_config_template = {
    "model": GenerationModel,
    "task": "triplet",
    "optimizer": "",
    "learning_rate": 1e-3,
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "use_bert_spc": True,
    "max_seq_len": 120,
    "patience": 99999,
    "sigma": 0.3,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,
    "use_amp": False,
    "overwrite_cache": False,
    "epochs": 100,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.0,
    "emb_dropout": 0.5,
    "num_layers": 1,
    "pooling": "avg",
    "gcn_dim": 300,
    "relation_constraint": True,
    "symmetry_decoding": False,
}

_usa_config_base = {
    "model": GenerationModel,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "yangheng/deberta-v3-base-absa-v1.1",
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "patience": 5,
    "use_bert_spc": True,
    "max_seq_len": 80,
    "SRD": 3,
    "dlcf_a": 2,  # the a in dlcf_dca_bert
    "dca_p": 1,  # the p in dlcf_dca_bert
    "dca_layer": 3,  # the layer in dlcf_dca_bert
    "use_syntax_based_SRD": False,
    "sigma": 0.3,
    "lcf": "cdw",
    "lsa": False,
    "window": "lr",
    "eta": 1,
    "eta_lr": 0.1,
    "dropout": 0,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 10,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,  # split train and test datasets into 5 folds and repeat 3 trainer
    "overwrite_cache": False,
}

_usa_config_english = {
    "model": GenerationModel,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "yangheng/deberta-v3-base-absa-v1.1",
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "patience": 99999,
    "use_bert_spc": True,
    "max_seq_len": 80,
    "SRD": 3,
    "dlcf_a": 2,  # the a in dlcf_dca_bert
    "dca_p": 1,  # the p in dlcf_dca_bert
    "dca_layer": 3,  # the layer in dlcf_dca_bert
    "use_syntax_based_SRD": False,
    "sigma": 0.3,
    "lcf": "cdw",
    "lsa": False,
    "window": "lr",
    "eta": 1,
    "eta_lr": 0.1,
    "dropout": 0.5,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 5,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,  # split train and test datasets into 5 folds and repeat 3 trainer
}

_usa_config_multilingual = {
    "model": GenerationModel,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "microsoft/mdeberta-v3-base",
    "use_bert_spc": True,
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "patience": 99999,
    "max_seq_len": 80,
    "SRD": 3,
    "dlcf_a": 2,  # the a in dlcf_dca_bert
    "dca_p": 1,  # the p in dlcf_dca_bert
    "dca_layer": 3,  # the layer in dlcf_dca_bert
    "use_syntax_based_SRD": False,
    "sigma": 0.3,
    "lcf": "cdw",
    "lsa": False,
    "window": "lr",
    "eta": 1,
    "eta_lr": 0.1,
    "dropout": 0.5,
    "l2reg": 0.000001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 5,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1
    # split train and test datasets into 5 folds and repeat 3 trainer
}

_usa_config_chinese = {
    "model": GenerationModel,
    "optimizer": "adamw",
    "learning_rate": 0.00002,
    "pretrained_bert": "bert-base-chinese",
    "use_bert_spc": True,
    "cache_dataset": True,
    "warmup_step": -1,
    "deep_ensemble": False,
    "patience": 99999,
    "max_seq_len": 80,
    "SRD": 3,
    "dlcf_a": 2,  # the a in dlcf_dca_bert
    "dca_p": 1,  # the p in dlcf_dca_bert
    "dca_layer": 3,  # the layer in dlcf_dca_bert
    "use_syntax_based_SRD": False,
    "sigma": 0.3,
    "lcf": "cdw",
    "lsa": False,
    "window": "lr",
    "eta": 1,
    "eta_lr": 0.1,
    "dropout": 0.5,
    "l2reg": 0.00001,
    "num_epoch": 10,
    "batch_size": 16,
    "initializer": "xavier_uniform_",
    "seed": 52,
    "output_dim": 3,
    "log_step": 5,
    "dynamic_truncate": True,
    "srd_alignment": True,  # for srd_alignment
    "evaluate_begin": 0,
    "similarity_threshold": 1,  # disable same text check for different examples
    "cross_validate_fold": -1,  # split train and test datasets into 5 folds and repeat 3 trainer
}


class USAConfigManager(ConfigManager):
    def __init__(self, args, **kwargs):
        """
        Available Params:   {'model': None,
                            'optimizer': "",
                            'learning_rate': 0.00002,
                            'pretrained_bert': "yangheng/deberta-v3-base-absa-v1.1",
                            'cache_dataset': True,
                            'warmup_step': -1,
                            'deep_ensemble': False,
                            'patience': 99999,
                            'use_bert_spc': True,
                            'max_seq_len': 80,
                            'SRD': 3,
                            'lsa': False,
                            'dlcf_a': 2,  # the a in dlcf_dca_bert
                            'dca_p': 1,  # the p in dlcf_dca_bert
                            'dca_layer': 3,  # the layer in dlcf_dca_bert
                            'use_syntax_based_SRD': False,
                            'sigma': 0.3,
                            'lcf': "cdw",
                            'window': "lr",
                            'eta': 1,
                            'eta_lr': 0.1,
                            'dropout': 0,
                            'l2reg': 0.000001,
                            'num_epoch': 10,
                            'batch_size': 16,
                            'initializer': 'xavier_uniform_',
                            'seed': {52, 214}
                            'output_dim': 3,
                            'log_step': 10,
                            'dynamic_truncate': True,
                            'srd_alignment': True,  # for srd_alignment
                            'evaluate_begin': 0,
                            'similarity_threshold': 1,  # disable same text check for different examples
                            'cross_validate_fold': -1   # split train and test datasets into 5 folds and repeat 3 trainer
                            }
        :param args:
        :param kwargs:
        """
        super().__init__(args, **kwargs)

    @staticmethod
    def set_usa_config(configType: str, newitem: dict):
        if isinstance(newitem, dict):
            if configType == "template":
                _usa_config_template.update(newitem)
            elif configType == "base":
                _usa_config_base.update(newitem)
            elif configType == "english":
                _usa_config_english.update(newitem)
            elif configType == "chinese":
                _usa_config_chinese.update(newitem)
            elif configType == "multilingual":
                _usa_config_multilingual.update(newitem)

            else:
                raise ValueError(
                    "Wrong value of configuration_class type supplied, please use one from following type: template, base, english, chinese, multilingual, glove, bert_baseline"
                )
        else:
            raise TypeError(
                "Wrong type of new configuration_class item supplied, please use dict e.g.{'NewConfig': NewValue}"
            )

    @staticmethod
    def set_usa_config_template(newitem):
        USAConfigManager.set_usa_config("template", newitem)

    @staticmethod
    def set_usa_config_base(newitem):
        USAConfigManager.set_usa_config("base", newitem)

    @staticmethod
    def set_usa_config_english(newitem):
        USAConfigManager.set_usa_config("english", newitem)

    @staticmethod
    def set_usa_config_chinese(newitem):
        USAConfigManager.set_usa_config("chinese", newitem)

    @staticmethod
    def set_usa_config_multilingual(newitem):
        USAConfigManager.set_usa_config("multilingual", newitem)

    @staticmethod
    def get_usa_config_template():
        _usa_config_template.update(_usa_config_template)
        return USAConfigManager(copy.deepcopy(_usa_config_template))

    @staticmethod
    def get_usa_config_base():
        _usa_config_template.update(_usa_config_base)
        return USAConfigManager(copy.deepcopy(_usa_config_template))

    @staticmethod
    def get_usa_config_english():
        _usa_config_template.update(_usa_config_english)
        return USAConfigManager(copy.deepcopy(_usa_config_template))

    @staticmethod
    def get_usa_config_chinese():
        _usa_config_template.update(_usa_config_chinese)
        return USAConfigManager(copy.deepcopy(_usa_config_template))

    @staticmethod
    def get_usa_config_multilingual():
        _usa_config_template.update(_usa_config_multilingual)
        return USAConfigManager(copy.deepcopy(_usa_config_template))
