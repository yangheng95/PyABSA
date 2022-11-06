# -*- coding: utf-8 -*-
# file: instructor_template.py
# time: 03/11/2022 13:21
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import math
import os
import pickle
import random

import numpy
import torch
from findfile import find_file, find_files
from torch.utils.data import DataLoader, random_split, ConcatDataset, RandomSampler, SequentialSampler
from transformers import BertModel

from pyabsa.framework.flag_class.flag_template import DeviceTypeOption

import pytorch_warmup as warmup

from pyabsa.utils.pyabsa_utils import print_args


class BaseTrainingInstructor:
    def __init__(self, config):
        """
        Initialize a trainer object template
        """
        if config.use_amp:
            try:
                self.scaler = torch.cuda.amp.GradScaler()
                print('Use AMP for trainer!')
            except Exception:
                self.scaler = None
        else:
            self.scaler = None

        random.seed(config.seed)
        numpy.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        config.device = torch.device(config.device)

        self.config = config
        self.logger = self.config.logger

        self.model = None
        self.tokenizer = None

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self.train_dataloaders = []
        self.valid_dataloaders = []
        self.test_dataloaders = []

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.optimizer = None
        self.initializer = None
        self.lr_scheduler = None
        self.warmup_scheduler = None

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _reload_model_state_dict(self, ckpt='./init_state_dict.bin'):
        if os.path.exists(ckpt):
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(torch.load(find_file(ckpt, or_key=['.bin', 'state_dict'])))
            else:
                self.model.load_state_dict(torch.load(find_file(ckpt, or_key=['.bin', 'state_dict'])))

    def _prepare_dataloader(self):
        if self.train_dataloader and self.valid_dataloader:
            self.valid_dataloaders = [self.valid_dataloader]
            self.train_dataloaders = [self.train_dataloader]

        elif self.config.cross_validate_fold < 1:
            train_sampler = RandomSampler(self.train_set if not self.train_set else self.train_set)
            self.train_dataloaders.append(DataLoader(dataset=self.train_set,
                                                     batch_size=self.config.batch_size,
                                                     sampler=train_sampler,
                                                     pin_memory=True))

            if self.valid_set and not self.valid_dataloader:
                valid_sampler = SequentialSampler(self.valid_set)
                self.valid_dataloader = DataLoader(dataset=self.valid_set,
                                                   batch_size=self.config.batch_size,
                                                   sampler=valid_sampler,
                                                   pin_memory=True)

            if self.test_set and not self.test_dataloader:
                test_sampler = SequentialSampler(self.test_set)
                self.test_dataloader = DataLoader(dataset=self.test_set,
                                                  batch_size=self.config.batch_size,
                                                  sampler=test_sampler,
                                                  pin_memory=True)

        else:
            split_dataset = self.train_set
            len_per_fold = len(split_dataset) // self.config.cross_validate_fold + 1
            folds = random_split(split_dataset, tuple([len_per_fold] * (self.config.cross_validate_fold - 1) + [
                len(split_dataset) - len_per_fold * (self.config.cross_validate_fold - 1)]))

            for f_idx in range(self.config.cross_validate_fold):
                train_set = ConcatDataset([x for i, x in enumerate(folds) if i != f_idx])
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set if not train_set else train_set)
                val_sampler = SequentialSampler(val_set if not val_set else val_set)
                self.train_dataloaders.append(
                    DataLoader(dataset=train_set, batch_size=self.config.batch_size, sampler=train_sampler))
                self.valid_dataloaders.append(
                    DataLoader(dataset=val_set, batch_size=self.config.batch_size, sampler=val_sampler))

    def _prepare_env(self):
        if os.path.exists('init_state_dict.bin'):
            os.remove('init_state_dict.bin')
        if self.config.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), 'init_state_dict.bin')

        # use DataParallel for trainer if device count larger than 1
        if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
            self.model.to(self.config.device)
            self.model = torch.nn.parallel.DataParallel(self.model).module
        else:
            self.model.to(self.config.device)

        self.config.device = torch.device(self.config.device)
        if self.config.device.type == 'cuda':
            self.logger.info("cuda memory allocated:{}".format(torch.cuda.memory_allocated(device=self.config.device)))

        print_args(self.config, self.logger)

    def _train(self, criterion):

        self._prepare_env()
        self._prepare_dataloader()
        self._resume_from_checkpoint()

        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(
                self.train_dataloaders[0]) * self.config.num_epoch)
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if len(self.valid_dataloaders) > 1:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)

    def _init_misc(self):
        raise NotImplementedError('Please implement this method in subclass')

    def _cache_or_load_dataset(self):
        raise NotImplementedError('Please implement this method in subclass')

    def _train_and_evaluate(self, criterion):
        raise NotImplementedError('Please implement this method in subclass')

    def _k_fold_train_and_evaluate(self, criterion):
        raise NotImplementedError('Please implement this method in subclass')

    def _evaluate_acc_f1(self, test_dataloader):
        raise NotImplementedError('Please implement this method in subclass')

    def _load_dataset_and_prepare_dataloader(self):
        raise NotImplementedError('Please implement this method in subclass')

    def _resume_from_checkpoint(self):
        logger = self.config.logger
        from_checkpoint_path = self.config.from_checkpoint
        if from_checkpoint_path:
            model_path = find_files(from_checkpoint_path, '.model')
            state_dict_path = find_files(from_checkpoint_path, '.state_dict')
            config_path = find_files(from_checkpoint_path, '.config')

            if from_checkpoint_path:
                if not config_path:
                    raise FileNotFoundError('.config file is missing!')
                config = pickle.load(open(config_path[0], 'rb'))
                if model_path:
                    if config.model != self.config.model:
                        logger.info('Warning, the checkpoint_class was not trained using {} from param_dict'.format(
                            self.config.model.__name__))
                    self.model = torch.load(model_path[0])
                if state_dict_path:
                    if torch.cuda.device_count() > 1 and self.config.device == DeviceTypeOption.ALL_CUDA:
                        self.model.module.load_state_dict(torch.load(state_dict_path[0]))
                    else:
                        self.model.load_state_dict(torch.load(state_dict_path[0]))
                    self.model.config = self.config
                    self.model.to(self.config.device)
                else:
                    logger.info('.model or .state_dict file is missing!')
            else:
                logger.info('No checkpoint_class found in {}'.format(from_checkpoint_path))
            logger.info('Resume trainer from Checkpoint: {}!'.format(from_checkpoint_path))
