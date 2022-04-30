# -*- coding: utf-8 -*-
# file: trainer.py
# time: 2021/4/22 0022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import copy
import os
import time

import findfile
import torch
import transformers
from torch import cuda
from transformers import AutoConfig

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
from pyabsa.utils.file_utils import query_local_version

from pyabsa.utils.logger import get_logger
from metric_visualizer import MetricVisualizer

from pyabsa.utils.pyabsa_utils import get_device

import warnings

warnings.filterwarnings('once')


def init_config(config, auto_device):
    config.device, config.device_name = get_device(auto_device)
    config.auto_device = auto_device
    config.device = 'cuda' if auto_device == 'allcuda' else config.device
    config.model_name = config.model.__name__.lower() if not isinstance(config.model, list) else 'ensemble'
    config.PyABSAVersion = __version__
    config.TransformersVersion = transformers.__version__
    config.TorchVersion = '{}+cuda{}'.format(torch.version.__version__, torch.version.cuda)

    return config


class Trainer:
    def __init__(self,
                 config: ConfigManager = None,
                 dataset=None,
                 from_checkpoint: str = None,
                 checkpoint_save_mode: int = 0,
                 auto_device=True,
                 path_to_save=None,
                 load_aug=False
                 ):
        """

        :param config: PyABSA.config.ConfigManager
        :param dataset: Dataset name, or a dataset_manager path, or a list of dataset_manager paths
        :param from_checkpoint: A checkpoint path to train based on
        :param checkpoint_save_mode: Save trained model to checkpoint,
                                     "checkpoint_save_mode=1" to save the state_dict,
                                     "checkpoint_save_mode=2" to save the whole model,
                                     "checkpoint_save_mode=3" to save the fine-tuned BERT,
                                     otherwise avoid saving checkpoint but return the trained model after training
        :param auto_device: True or False, otherwise 'allcuda', 'cuda:1', 'cpu' works
        :param path_to_save=None: Specify path to save checkpoints
        :param load_aug=False: Load the available augmentation dataset if any

        """
        if not torch.cuda.device_count() > 1 and auto_device == 'allcuda':
            print('Cuda count <= 1, reset auto_device=True')
            auto_device = True
        if 'hidden_dim' not in config.args or 'embed_dim' not in config.args:
            pretrain_config = AutoConfig.from_pretrained(config.pretrained_bert)
            config.hidden_dim = pretrain_config.hidden_size
            config.embed_dim = pretrain_config.hidden_size
        config.ABSADatasetsVersion = query_local_version()
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
            self.config.dataset_name = dataset.dataset_name
        else:
            dataset = DatasetItem('custom_dataset', dataset)
            self.config.dataset_name = dataset.dataset_name
        self.dataset_file = detect_dataset(dataset, task=self.task, load_aug=load_aug)
        self.config.dataset_file = self.dataset_file

        self.config = init_config(self.config, auto_device)
        if 'MV' not in self.config.args:
            self.config.MV = MetricVisualizer(name=config.model.__name__ + '-' + self.config.dataset_name,
                                              trial_tag='Model & Dataset',
                                              trial_tag_list=[config.model.__name__ + '-' + self.config.dataset_name])

        # self.config.ETA_MV = MetricVisualizer('eta-' + self.config.model.__name__ + '-' + self.config.dataset_name, trial_tag='Model & Dataset')

        self.from_checkpoint = findfile.find_dir(os.getcwd(), from_checkpoint) if from_checkpoint else ''
        self.checkpoint_save_mode = checkpoint_save_mode
        self.config.save_mode = checkpoint_save_mode
        log_name = self.config.model_name
        self.logger = get_logger(os.getcwd(), log_name=log_name, log_type='training')

        if checkpoint_save_mode or self.dataset_file['valid']:
            if path_to_save:
                config.model_path_to_save = path_to_save
            elif self.dataset_file['valid'] and not checkpoint_save_mode:
                print('Using Validation set needs to save checkpoint, turn on checkpoint-saving ...')
                config.model_path_to_save = 'checkpoints'
                self.config.save_mode = 1
            else:
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
        model = None
        for i, s in enumerate(seeds):
            self.config.seed = s
            if self.checkpoint_save_mode:
                model_path.append(self.train_func(self.config, self.from_checkpoint, self.logger))
            else:
                # always return the last trained model if dont save trained model
                model = self.model_class(model_arg=self.train_func(self.config, self.from_checkpoint, self.logger))
        self.config.seed = seeds

        # save_path = '{}_{}'.format(self.config.model_name, self.config.dataset_name)
        # self.config.MV.summary(save_path)

        while self.logger.handlers:
            self.logger.removeHandler(self.logger.handlers[0])

        if self.checkpoint_save_mode:
            if os.path.exists(max(model_path)):
                self.inference_model = self.model_class(max(model_path))
        else:
            self.inference_model = model

    def load_trained_model(self):
        if not self.inference_model:
            print('No trained model found, this could happen while training only using training set.')
        self.inference_model.to(self.config.device)
        return self.inference_model

    def destroy(self):
        del self.inference_model
        cuda.empty_cache()
        time.sleep(3)


class APCTrainer(Trainer):
    pass


class ATEPCTrainer(Trainer):
    pass


class TextClassificationTrainer(Trainer):
    pass
