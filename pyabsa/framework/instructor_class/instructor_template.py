# -*- coding: utf-8 -*-
# file: instructor_template.py
# time: 03/11/2022 13:21
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

import math
import os
import pickle
import random
import re
from hashlib import sha256

import numpy
import pytorch_warmup as warmup
import torch
from findfile import find_file, find_files
from termcolor import colored
from torch.utils.data import (
    DataLoader,
    random_split,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
)
from transformers import BertModel

from pyabsa.framework.flag_class.flag_template import DeviceTypeOption
from pyabsa.framework.sampler_class.imblanced_sampler import ImbalancedDatasetSampler
from pyabsa.utils.pyabsa_utils import print_args, fprint


class BaseTrainingInstructor:
    def __init__(self, config):
        """
        Initialize a trainer object template
        """
        # Check if mixed precision training is enabled
        if config.use_amp:
            try:
                # Initialize AMP grad scaler if available
                self.scaler = torch.cuda.amp.GradScaler()
                # Print a message to inform the user that AMP is being used
                fprint(
                    colored(
                        "Use torch.AMP for training! Please disable it if you encounter convergence problems.",
                        "yellow",
                    )
                )
            except Exception:
                # If AMP is not available, set scaler to None
                self.scaler = None
        else:
            # If AMP is not enabled, set scaler to None
            self.scaler = None

        # Set random seed for reproducibility
        random.seed(config.seed)
        numpy.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)

        # Set config, logger, model, tokenizer, and dataloaders to None
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

        # Set train, validation, and test sets to None
        self.train_set = None
        self.valid_set = None
        self.test_set = None

        # Set optimizer, initializer, lr_scheduler, warmup_scheduler, tokenizer, and embedding_matrix to None
        self.optimizer = None
        self.initializer = None
        self.lr_scheduler = None
        self.warmup_scheduler = None
        self.tokenizer = None
        self.embedding_matrix = None

    def _reset_params(self):
        """
        Reset the parameters of the model before training.
        """
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.initializer(p)
                        else:
                            stdv = 1.0 / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _reload_model_state_dict(self, ckpt="./init_state_dict.bin"):
        """
        Reload the model state dictionary from a checkpoint file.
        :param ckpt: The path to the checkpoint file.
        """
        if os.path.exists(ckpt):
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(
                    torch.load(find_file(ckpt, or_key=[".bin", "state_dict"]))
                )
            else:
                self.model.load_state_dict(
                    torch.load(find_file(ckpt, or_key=[".bin", "state_dict"]))
                )

    def load_cache_dataset(self, **kwargs):
        """
        Load the dataset from cache if it exists and not set to overwrite the cache. Otherwise, return None.
        :param kwargs: Additional keyword arguments.
        :return: The path to the cache file if it exists. Otherwise, return None.
        """
        # Generate the hash tag for the cache file
        config_str = re.sub(
            r"<.*?>",
            "",
            str(
                sorted(
                    [str(self.config.args[k]) for k in self.config.args if k != "seed"]
                )
            ),
        )
        hash_tag = sha256(config_str.encode()).hexdigest()

        # Construct the path to the cache file
        cache_path = "{}.{}.dataset.{}.cache".format(
            self.config.model_name, self.config.dataset_name, hash_tag
        )

        # Load the dataset from cache if it exists and not set to overwrite the cache
        if os.path.exists(cache_path) and not self.config.overwrite_cache:
            with open(cache_path, mode="rb") as f_cache:
                self.config.logger.info("Load cache dataset from {}".format(cache_path))
                (
                    self.train_set,
                    self.valid_set,
                    self.test_set,
                    self.config,
                ) = pickle.load(f_cache)
                _config = kwargs.get("config", None)
                if _config:
                    _config.update(self.config)
                    _config.args_call_count.update(self.config.args_call_count)
                return cache_path

        return cache_path

    def save_cache_dataset(self, cache_path=None, **kwargs):
        """
        Save the dataset to cache for faster loading in the future.
        :param kwargs: Additional arguments for saving the dataset cache.
        :param cache_path: The path to the cache file.
        :return: The path to the saved cache file.
        """
        if cache_path is None:
            config_str = re.sub(
                r"<.*?>",
                "",
                str(
                    sorted(
                        [
                            str(self.config.args[k])
                            for k in self.config.args
                            if k != "seed"
                        ]
                    )
                ),
            )
            hash_tag = sha256(config_str.encode()).hexdigest()
            cache_path = "{}.{}.dataset.{}.cache".format(
                self.config.model_name, self.config.dataset_name, hash_tag
            )
        if (
            not os.path.exists(cache_path) or self.config.overwrite_cache
        ) and self.config.cache_dataset:
            with open(cache_path, mode="wb") as f_cache:
                self.config.logger.info("Save cache dataset to {}".format(cache_path))
                pickle.dump(
                    [self.train_set, self.valid_set, self.test_set, self.config],
                    f_cache,
                )
                return cache_path
        return None

    def _prepare_dataloader(self):
        """
        Prepares the data loaders for training, validation, and testing.
        """
        if self.config.get("train_sampler", "random") == "random":
            train_sampler = RandomSampler(self.train_set)
        elif self.config.get("train_sampler", "random") == "imbalanced":
            train_sampler = ImbalancedDatasetSampler(self.train_set)
        elif self.config.get("train_sampler", "sequential") == "sequential":
            train_sampler = SequentialSampler(self.train_set)
        else:
            raise ValueError(
                "train_sampler should be in [random, imbalanced, sequential]"
            )

        # If both training and validation dataloaders are already set, use them as is
        if self.train_dataloader and self.valid_dataloader:
            self.valid_dataloaders = [self.valid_dataloader]
            self.train_dataloaders = [self.train_dataloader]

        # Otherwise, set up training, validation, and testing dataloaders based on the configuration
        elif self.config.cross_validate_fold < 1:
            # Single dataset, no cross-validation
            self.train_dataloaders.append(
                DataLoader(
                    dataset=self.train_set,
                    batch_size=self.config.batch_size,
                    sampler=train_sampler,
                    pin_memory=True,
                )
            )

            # Set up the validation dataloader
            if self.valid_set and not self.valid_dataloader:
                valid_sampler = SequentialSampler(self.valid_set)
                self.valid_dataloader = DataLoader(
                    dataset=self.valid_set,
                    batch_size=self.config.batch_size,
                    sampler=valid_sampler,
                    pin_memory=True,
                )

        # Cross-validation
        else:
            split_dataset = self.train_set
            len_per_fold = len(split_dataset) // self.config.cross_validate_fold + 1
            # Split the dataset into folds
            folds = random_split(
                split_dataset,
                tuple(
                    [len_per_fold] * (self.config.cross_validate_fold - 1)
                    + [
                        len(split_dataset)
                        - len_per_fold * (self.config.cross_validate_fold - 1)
                    ]
                ),
            )

            # Set up dataloaders for each fold
            for f_idx in range(self.config.cross_validate_fold):
                train_set = ConcatDataset(
                    [x for i, x in enumerate(folds) if i != f_idx]
                )
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set)
                val_sampler = SequentialSampler(val_set)
                self.train_dataloaders.append(
                    DataLoader(
                        dataset=train_set,
                        batch_size=self.config.batch_size,
                        sampler=train_sampler,
                    )
                )
                self.valid_dataloaders.append(
                    DataLoader(
                        dataset=val_set,
                        batch_size=self.config.batch_size,
                        sampler=val_sampler,
                    )
                )

        # Set up the testing dataloader
        if self.test_set and not self.test_dataloader:
            test_sampler = SequentialSampler(self.test_set)
            self.test_dataloader = DataLoader(
                dataset=self.test_set,
                batch_size=self.config.batch_size,
                sampler=test_sampler,
                pin_memory=True,
            )

    def _prepare_env(self):
        """
        Prepares the environment for training, including setting the tokenizer and embedding matrix,
        removing the initial state dictionary file if it exists, and setting up the model on the appropriate device.
        """
        # Set the tokenizer and embedding matrix
        self.config.tokenizer = self.tokenizer
        self.config.embedding_matrix = self.embedding_matrix

        # Remove initial state dictionary file if it exists
        if os.path.exists("init_state_dict.bin"):
            os.remove("init_state_dict.bin")

        # Save the model state dict to initial state dictionary file if using k-fold cross-validation
        if self.config.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), "init_state_dict.bin")

        # Use DataParallel for trainer if device count larger than 1
        if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
            self.model.to(self.config.device)
            self.model = torch.nn.parallel.DataParallel(self.model).module
        else:
            self.model.to(self.config.device)

        # Set the device and print CUDA memory if applicable
        self.config.device = torch.device(self.config.device)
        if self.config.device.type == DeviceTypeOption.CUDA:
            self.logger.info(
                "cuda memory allocated:{}".format(
                    torch.cuda.memory_allocated(device=self.config.device)
                )
            )

        # Print the model architecture and arguments
        if self.config.get("verbose", True):
            self.config.logger.info(
                "Model Architecture:\n {}".format(self.model.__repr__())
            )
        print_args(self.config, self.logger)

    def _train(self, criterion):
        """
        Train the model on a given criterion.

        Args:
            criterion: The loss function used to train the model.

        Returns:
            If there is only one validation dataloader, return the training results.
            If there are more than one validation dataloaders, perform k-fold cross-validation and return the results.
        """
        # Prepare the environment and dataloader
        self._prepare_env()
        self._prepare_dataloader()

        # Compile the model using torch v2.0.0+ compile feature if specified
        if self.config.get("use_torch_compile", False):
            try:
                self.model = torch.compile(self.model)
                self.logger.info("use torch v2.0+ compile feature")
            except Exception as e:
                pass

        # Resume training from a previously trained model
        self._resume_from_checkpoint()

        # Initialize the learning rate scheduler if warmup_step is specified
        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloaders[0]) * self.config.num_epoch,
            )
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        # Perform k-fold cross-validation if there are multiple validation dataloaders
        if len(self.valid_dataloaders) > 1:
            return self._k_fold_train_and_evaluate(criterion)
        # Train and evaluate the model if there is only one validation dataloader
        else:
            return self._train_and_evaluate(criterion)

    def _init_misc(self):
        """
        Initialize miscellaneous settings specific to the subclass implementation.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _cache_or_load_dataset(self):
        """
        Cache or load the dataset.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _train_and_evaluate(self, criterion):
        """
        Train and evaluate the model.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _k_fold_train_and_evaluate(self, criterion):
        """
        Train and evaluate the model using k-fold cross validation.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _evaluate_acc_f1(self, test_dataloader):
        """
        Evaluate the accuracy and F1 score of the model.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _load_dataset_and_prepare_dataloader(self):
        """
        Load the dataset and prepare the dataloader.
        This method should be implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _resume_from_checkpoint(self):
        """
        Resumes training from a checkpoint if a valid checkpoint path is provided in the configuration file,
        by loading the model, state dictionary, and configuration from the checkpoint files.
        """
        logger = self.config.logger
        from_checkpoint_path = get_resume_checkpoint(self.config)
        if from_checkpoint_path:
            # Get model, state dict, and configuration paths from checkpoint path
            model_path = find_files(from_checkpoint_path, ".model")
            state_dict_path = find_files(from_checkpoint_path, ".state_dict")
            config_path = find_files(from_checkpoint_path, ".config")

            if from_checkpoint_path:
                # Check if configuration file exists in the checkpoint directory
                if not config_path:
                    raise FileNotFoundError(".config file is missing!")
                # Load configuration file from checkpoint directory
                config = pickle.load(open(config_path[0], "rb"))
                # Load model from checkpoint directory
                if model_path:
                    # Check if the model in the checkpoint was trained using the same parameters as the current model
                    if config.model != self.config.model:
                        logger.info(
                            "Warning, the checkpoint was not trained using {} from param_dict".format(
                                self.config.model.__name__
                            )
                        )
                    self.model = torch.load(model_path[0])
                # Load state dictionary from checkpoint directory
                if state_dict_path:
                    if (
                        torch.cuda.device_count() > 1
                        and self.config.device == DeviceTypeOption.ALL_CUDA
                    ):
                        self.model.module.load_state_dict(
                            torch.load(state_dict_path[0])
                        )
                    else:
                        self.model.load_state_dict(
                            torch.load(
                                state_dict_path[0], map_location=self.config.device
                            ),
                            strict=False,
                        )
                    self.model.config = self.config
                    self.model.to(self.config.device)
                else:
                    logger.info(".model or .state_dict file is missing!")
            else:
                logger.info("No checkpoint found in {}".format(from_checkpoint_path))
            logger.info(
                "Resume trainer from Checkpoint: {}!".format(from_checkpoint_path)
            )


def get_resume_checkpoint(config):
    from pyabsa.framework.checkpoint_class.checkpoint_template import CheckpointManager

    # Extract the path to the checkpoint from the config object
    ckpt = config.from_checkpoint
    if config.from_checkpoint:
        # Look for the config file in the checkpoint directory
        config_path = find_files(config.from_checkpoint, ".config")

        if not config_path:
            # If the config file is not found, try to parse the checkpoint file
            try:
                ckpt = CheckpointManager().parse_checkpoint(
                    checkpoint=config.from_checkpoint, task_code=config.task_code
                )
                config.logger.info("Checkpoint downloaded at: {}".format(ckpt))
            except Exception as e:
                fprint(e)
                # If parsing the checkpoint file fails, raise an error
                raise ValueError(
                    "Cannot find checkpoint file in {}".format(config.from_checkpoint)
                )
    # Return the path to the checkpoint file
    return ckpt
