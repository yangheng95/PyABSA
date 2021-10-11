# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import copy
import os

import findfile
from findfile import find_dir

from pyabsa import __version__

from pyabsa.functional.dataset import DatasetItem
from pyabsa.functional.config.config_manager import ConfigManager
from pyabsa.functional.dataset import detect_dataset
from pyabsa.core.apc.prediction.sentiment_classifier import SentimentClassifier
from pyabsa.core.apc.training.apc_trainer import train4apc
from pyabsa.core.atepc.prediction.aspect_extractor import AspectExtractor
from pyabsa.core.atepc.training.atepc_trainer import train4atepc
from pyabsa.core.tc.prediction.text_classifier import TextClassifier
from pyabsa.core.tc.training.classifier_trainer import train4classification

from pyabsa.functional.config.apc_config_manager import APCConfigManager
from pyabsa.functional.config.atepc_config_manager import ATEPCConfigManager
from pyabsa.functional.config.classification_config_manager import ClassificationConfigManager
from pyabsa.functional.dataset import ABSADatasetList

from pyabsa.utils.logger import get_logger

from pyabsa.utils.pyabsa_utils import get_device


def init_config(config, auto_device):
    config.device, config.device_name = get_device(auto_device)

    config.model_name = config.model.__name__.lower()
    config.Version = __version__

    if 'use_syntax_based_SRD' in config:
        print('-' * 130)
        print('Force to use syntax distance-based semantic-relative distance,'
              ' however Chinese is not supported to parse syntax distance yet!  ')
        print('-' * 130)
    return config


class Trainer:
    def __init__(self,
                 config: ConfigManager = None,
                 dataset: str = None,
                 from_checkpoint: str = '',
                 checkpoint_save_mode: int = 1,
                 auto_device=True):
        """

        :param config: PyABSA.config.ConfigManager
        :param dataset: Dataset name, or a dataset_manager path, or a list of dataset_manager paths
        :param from_checkpoint: A checkpoint path to train based on
        :param checkpoint_save_mode: Save trained model to checkpoint,
                                     "checkpoint_save_mode=1" to save the state_dict,
                                     "checkpoint_save_mode=2" to save the whole model,
                                     "checkpoint_save_mode=3" to save the fine-tuned BERT,
                                     otherwise avoid to save checkpoint but return the trained model after training
        :param auto_device: True or False, otherwise 'cuda', 'cpu' works

        """
        if isinstance(config, APCConfigManager):
            self.train_func = train4apc
            self.model_class = SentimentClassifier
            self.task = 'apc'
        elif isinstance(config, ATEPCConfigManager):
            self.train_func = train4atepc
            self.model_class = AspectExtractor
            self.task = 'atepc'
        elif isinstance(config, ClassificationConfigManager):
            self.train_func = train4classification
            self.model_class = TextClassifier
            self.task = 'classification'

        self.config = config
        if isinstance(dataset, DatasetItem):
            self.config.dataset_item = list(dataset)
            self.config.dataset_name = dataset.dataset_name
        else:
            custom_dataset = DatasetItem('custom_dataset', dataset)
            self.config.dataset_item = list(custom_dataset)
            self.config.dataset_name = custom_dataset.dataset_name
        self.dataset_file = detect_dataset(dataset, task=self.task)
        self.config.dataset_file = self.dataset_file
        self.config = init_config(self.config, auto_device)

        self.from_checkpoint = findfile.find_dir(os.getcwd(), from_checkpoint) if from_checkpoint else ''
        self.checkpoint_save_mode = checkpoint_save_mode
        self.config.save_mode = checkpoint_save_mode
        log_name = self.config.model_name
        self.logger = get_logger(os.getcwd(), log_name=log_name, log_type='training')

        if checkpoint_save_mode:
            config.model_path_to_save = 'checkpoints'
        else:
            config.model_path_to_save = None

        self.inference_model = None

        self.train()

    def train(self):
        """
        just return the trained model for inference (e.g., polarity classification, aspect-term extraction)
        """

        if isinstance(self.config.seed, int):
            self.config.seed = [self.config.seed]

        model_path = []
        seeds = self.config.seed
        for _, s in enumerate(seeds):
            config = copy.deepcopy(self.config)
            config.seed = s
            if self.checkpoint_save_mode:
                model_path.append(self.train_func(config, self.from_checkpoint, self.logger))
            else:
                # always return the last trained model if dont save trained model
                model = self.model_class(model_arg=self.train_func(config, self.from_checkpoint, self.logger))
        while self.logger.handlers:
            self.logger.removeHandler(self.logger.handlers[0])

        if self.checkpoint_save_mode:
            self.inference_model = self.model_class(max(model_path))
        else:
            self.inference_model = model

    def load_trained_model(self):
        self.inference_model.to(self.config.device)
        return self.inference_model


class APCTrainer(Trainer):
    pass


class ATEPCTrainer(Trainer):
    pass


class TextClassificationTrainer(Trainer):
    pass
