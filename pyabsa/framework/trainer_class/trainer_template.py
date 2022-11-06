# -*- coding: utf-8 -*-
# file: trainer.py
# time: 02/11/2022 21:15
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.
# Copyright (C) 2021. All Rights Reserved.
import os
import time
from pathlib import Path
from typing import Union

import findfile
import torch
import transformers
from metric_visualizer import MetricVisualizer
from torch import cuda
from transformers import AutoConfig

from pyabsa import __version__ as PyABSAVersion
from ..configuration_class.config_verification import config_check
from ..dataset_class.dataset_dict_class import DatasetDict

from ..flag_class.flag_template import DeviceTypeOption, ModelSaveOption
from ..configuration_class.configuration_template import ConfigManager

from pyabsa.utils.logger.logger import get_logger

from pyabsa.utils.pyabsa_utils import get_device
from ...utils.check_utils import query_local_datasets_version
from ...utils.data_utils.dataset_item import DatasetItem
from ...utils.data_utils.dataset_manager import detect_dataset

import warnings

warnings.filterwarnings('once')


def init_config(config):
    get_device(config)
    if not torch.cuda.device_count() > 1 and config.auto_device == DeviceTypeOption.ALL_CUDA:
        print('Cuda devices count <= 1, so reset auto_device=True to auto specify device...')
        config.auto_device = True
    config.model_name = config.model.__name__.lower() if not isinstance(config.model, list) else 'ensemble_model'

    if config.get('pretrained_bert', None):
        try:
            pretrain_config = AutoConfig.from_pretrained(config.pretrained_bert)
            config.hidden_dim = pretrain_config.hidden_size
            config.embed_dim = pretrain_config.hidden_size
        except:
            pass
    elif not config.get('hidden_dim', None) or not config.get('embed_dim', None):
        if config.get('hidden_dim', None):
            config.embed_dim = config.hidden_dim
        elif config.get('embed_dim', None):
            config.hidden_dim = config.embed_dim
        else:
            config.hidden_dim = 768
            config.embed_dim = 768

    config.ABSADatasetsVersion = query_local_datasets_version()
    config.PyABSAVersion = PyABSAVersion
    config.TransformersVersion = transformers.__version__
    config.TorchVersion = '{}+cuda{}'.format(torch.version.__version__, torch.version.cuda)

    if isinstance(config.dataset, DatasetItem):
        config.dataset_name = config.dataset.dataset_name
    else:
        dataset = DatasetItem('custom_dataset', config.dataset)
        config.dataset_name = dataset.dataset_name

    if 'MV' not in config.args:
        config.MV = MetricVisualizer(name=config.model.__name__ + '-' + config.dataset_name,
                                     trial_tag='Model & Dataset',
                                     trial_tag_list=[config.model.__name__ + '-' + config.dataset_name])

    config.from_checkpoint = findfile.find_dir(os.getcwd(), config.from_checkpoint) if config.from_checkpoint else ''
    checkpoint_save_mode = config.checkpoint_save_mode

    config.save_mode = checkpoint_save_mode
    config_check(config)
    config.logger = get_logger(os.getcwd(), log_name=config.model_name, log_type='trainer')
    config.logger.info('PyABSA version: {}'.format(config.PyABSAVersion))
    config.logger.info('Transformers version: {}'.format(config.TransformersVersion))
    config.logger.info('Torch version: {}'.format(config.TorchVersion))
    config.logger.info('Device: {}'.format(config.device_name))

    return config


class Trainer:
    def __init__(self,
                 config: ConfigManager = None,
                 dataset: Union[DatasetItem, Path, str, DatasetDict] = None,
                 from_checkpoint: Union[Path, str] = None,
                 checkpoint_save_mode: Union[ModelSaveOption, int] = ModelSaveOption.SAVE_MODEL_STATE_DICT,
                 auto_device: Union[str, bool] = DeviceTypeOption.AUTO,
                 path_to_save: Union[Path, str] = None,
                 load_aug=False
                 ):
        """
        Init a trainer for trainer a APC, ATEPC, TC or TAD model, after trainer,
        you need to call load_trained_model() to get the trained model for inference.

        :param config: PyABSA.config.ConfigManager
        :param dataset: Dataset name, or a dataset_manager path, or a list of dataset_manager paths
        :param from_checkpoint: A checkpoint path to train based on
        :param checkpoint_save_mode: Save trained model to checkpoint,
                                     "checkpoint_save_mode=1" to save the state_dict,
                                     "checkpoint_save_mode=2" to save the whole model,
                                     "checkpoint_save_mode=3" to save the fine-tuned BERT,
                                     otherwise avoid saving checkpoint but return the trained model after trainer
        :param auto_device: True or False, otherwise 'allcuda', 'cuda:1', 'cpu' works
        :param path_to_save=None: Specify path to save checkpoints
        :param load_aug=False: Load the available augmentation dataset if any

        """

        # device check

        self.config = config
        self.config.dataset = dataset
        self.config.from_checkpoint = from_checkpoint
        self.config.checkpoint_save_mode = checkpoint_save_mode
        self.config.auto_device = auto_device
        self.config.path_to_save = path_to_save
        self.config.load_aug = load_aug
        self.config.inference_model = None

        self.config = init_config(self.config)

        self.config.task_code = None
        self.config.task_name = None

        self.training_instructor = None
        self.inference_model_class = None
        self.inference_model = None

    def _run(self):
        """
        just return the trained model for inference (e.g., polarity classification, aspect-term extraction)
        """
        if isinstance(self.config.dataset, DatasetDict):
            self.config.dataset_dict = self.config.dataset
            self.config.dataset = self.config.dataset_dict['dataset_name']
            self.config.dataset_file = {'train': [], 'valid': [], 'test': []}
        else:
            # detect dataset
            dataset_file = detect_dataset(self.config.dataset, task_code=self.config.task_code, load_aug=self.config.load_aug, config=self.config)
            self.config.dataset_file = dataset_file
        if self.config.checkpoint_save_mode or self.config.dataset_file['valid'] or self.config.dataset_dict['test']:
            if self.config.path_to_save:
                self.config.model_path_to_save = self.config.path_to_save
            elif self.config.dataset_file['valid'] and not self.config.checkpoint_save_mode:
                print('Using Validation set needs to save checkpoint, turn on checkpoint-saving ...')
                self.config.model_path_to_save = 'checkpoints'
                self.config.save_mode = 1
            else:
                self.config.model_path_to_save = 'checkpoints'
        else:
            self.config.model_path_to_save = None

        # set random seed
        if isinstance(self.config.seed, int):
            self.config.seed = [self.config.seed]
        seeds = self.config.seed

        # trainer using all random seeds
        model_path = []
        model = None
        for i, s in enumerate(seeds):
            self.config.seed = s
            if self.config.checkpoint_save_mode:
                model_path.append(self.training_instructor(self.config).run())
            else:
                # always return the last trained model if you don't save trained model
                model = self.inference_model_class(checkpoint=self.training_instructor(self.config).run())
        self.config.seed = seeds

        # remove logger
        while self.config.logger.handlers:
            self.config.logger.removeHandler(self.config.logger.handlers[0])

        # load inference model
        if self.config.checkpoint_save_mode:
            if os.path.exists(max(model_path)):
                self.inference_model = self.inference_model_class(max(model_path))
        else:
            self.inference_model = model

    def load_trained_model(self):
        """
        Load trained model for inference
        """
        if not self.inference_model:
            print('No trained model found, this could happen while trainer only using trainer set.')
        self.inference_model.to(self.config.device)
        return self.inference_model

    def destroy(self):
        del self.inference_model
        cuda.empty_cache()
        time.sleep(3)
