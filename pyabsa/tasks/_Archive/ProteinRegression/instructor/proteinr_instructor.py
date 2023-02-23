# -*- coding: utf-8 -*-
# file: classifier_instructor.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import time

import numpy
import numpy as np
import torch
import torch.nn as nn
from findfile import find_file
from sklearn import metrics
from torch import cuda
from torch.utils.data import (
    DataLoader,
    random_split,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from pyabsa.networks.losses.R2Loss import R2Loss
from pyabsa.utils.file_utils.file_utils import save_model
from ..dataset_utils.__classic__.data_utils_for_training import GloVeProteinRDataset
from ..dataset_utils.__plm__.data_utils_for_training import BERTProteinRDataset
from ..models import GloVeProteinRModelList, BERTProteinRModelList
from pyabsa.utils.pyabsa_utils import init_optimizer, fprint

import pytorch_warmup as warmup

from pyabsa.framework.tokenizer_class.tokenizer_class import (
    Tokenizer,
    build_embedding_matrix,
    PretrainedTokenizer,
)


class ProteinRTrainingInstructor(BaseTrainingInstructor):
    def _init_misc(self):
        # use DataParallel for trainer if device count larger than 1
        if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
            self.model.to(self.config.device)
            self.model = torch.nn.parallel.DataParallel(self.model).module
        else:
            self.model.to(self.config.device)

        self.optimizer = init_optimizer(self.config.optimizer)(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2reg,
        )

        self.train_dataloaders = []
        self.valid_dataloaders = []

        if os.path.exists("./init_state_dict.bin"):
            os.remove("./init_state_dict.bin")
        if self.config.cross_validate_fold > 0:
            torch.save(self.model.state_dict(), "./init_state_dict.bin")

        self.config.device = torch.device(self.config.device)
        if self.config.device.type == DeviceTypeOption.CUDA:
            self.logger.info(
                "cuda memory allocated:{}".format(
                    torch.cuda.memory_allocated(device=self.config.device)
                )
            )

    def _cache_or_load_dataset(self):
        pass

    def _evaluate_acc_f1(self, test_dataloader):
        pass

    def _load_dataset_and_prepare_dataloader(self):
        self.config.inputs_cols = self.config.model.inputs

        cache_path = self.load_cache_dataset()

        # init BERT-based model and dataset
        if hasattr(BERTProteinRModelList, self.config.model.__name__):
            self.tokenizer = PretrainedTokenizer(self.config)
            if cache_path is None or self.config.overwrite_cache:
                self.train_set = BERTProteinRDataset(
                    self.config, self.tokenizer, dataset_type="train"
                )
                self.test_set = BERTProteinRDataset(
                    self.config, self.tokenizer, dataset_type="test"
                )
                self.valid_set = BERTProteinRDataset(
                    self.config, self.tokenizer, dataset_type="valid"
                )

            try:
                self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
            except ValueError as e:
                fprint("Init pretrained model failed, exception: {}".format(e))

            # init the model behind the construction of datasets in case of updating output_dim
            self.model = self.config.model(self.bert, self.config).to(
                self.config.device
            )

        elif hasattr(GloVeProteinRModelList, self.config.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = Tokenizer.build_tokenizer(
                config=self.config,
                cache_path="{0}_tokenizer.dat".format(
                    os.path.basename(self.config.dataset_name)
                ),
                pre_tokenizer=AutoTokenizer.from_pretrained(
                    self.config.pretrained_bert
                ),
            )
            self.embedding_matrix = build_embedding_matrix(
                config=self.config,
                tokenizer=self.tokenizer,
                cache_path="{0}_{1}_embedding_matrix.dat".format(
                    str(self.config.embed_dim),
                    os.path.basename(self.config.dataset_name),
                ),
            )
            self.train_set = GloVeProteinRDataset(
                self.config, self.tokenizer, dataset_type="train"
            )
            self.test_set = GloVeProteinRDataset(
                self.config, self.tokenizer, dataset_type="test"
            )
            self.valid_set = GloVeProteinRDataset(
                self.config, self.tokenizer, dataset_type="valid"
            )

            self.model = self.config.model(self.embedding_matrix, self.config).to(
                self.config.device
            )
            self.config.embedding_matrix = self.embedding_matrix

        self.config.tokenizer = self.tokenizer
        self.save_cache_dataset()

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def reload_model(self, ckpt="./init_state_dict.bin"):
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(find_file(ckpt, or_key=[".bin", "state_dict"]))
            )

    def _prepare_dataloader(self):
        if self.config.cross_validate_fold < 1:
            train_sampler = RandomSampler(
                self.train_set if not self.train_set else self.train_set
            )
            self.train_dataloaders.append(
                DataLoader(
                    dataset=self.train_set,
                    batch_size=self.config.batch_size,
                    sampler=train_sampler,
                    pin_memory=True,
                )
            )
            if self.test_set:
                self.test_dataloader = DataLoader(
                    dataset=self.test_set,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                )

            if self.valid_set:
                self.valid_dataloader = DataLoader(
                    dataset=self.valid_set,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                )
        else:
            split_dataset = self.train_set
            len_per_fold = len(split_dataset) // self.config.cross_validate_fold + 1
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

            for f_idx in range(self.config.cross_validate_fold):
                train_set = ConcatDataset(
                    [x for i, x in enumerate(folds) if i != f_idx]
                )
                val_set = folds[f_idx]
                train_sampler = RandomSampler(train_set if not train_set else train_set)
                val_sampler = SequentialSampler(val_set if not val_set else val_set)
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
                if self.test_set:
                    self.test_dataloader = DataLoader(
                        dataset=self.test_set,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                    )

    def _train(self, criterion):
        self._prepare_dataloader()

        if self.config.warmup_step >= 0:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloaders[0]) * self.config.num_epoch,
            )
            self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        if self.valid_dataloaders:
            return self._k_fold_train_and_evaluate(criterion)
        else:
            return self._train_and_evaluate(criterion)

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_r2 = -torch.inf
        save_path = "{0}/{1}_{2}".format(
            self.config.model_path_to_save,
            self.config.model_name,
            self.config.dataset_name,
        )

        losses = []

        self.config.metrics_of_this_checkpoint = {"r2": 0}
        self.config.max_test_metrics = {"max_test_r2": 0}

        self.logger.info(
            "***** Running training for {} *****".format(self.config.task_name)
        )
        self.logger.info("Training set examples = %d", len(self.train_set))
        if self.valid_set:
            self.logger.info("Valid set examples = %d", len(self.valid_set))
        if self.test_set:
            self.logger.info("Test set examples = %d", len(self.test_set))
        self.logger.info("Batch size = %d", self.config.batch_size)
        self.logger.info(
            "Num steps = %d",
            len(self.train_dataloaders[0])
            // self.config.batch_size
            * self.config.num_epoch,
        )
        patience = self.config.patience + self.config.evaluate_begin
        if self.config.log_step < 0:
            self.config.log_step = (
                len(self.train_dataloaders[0])
                if self.config.log_step < 0
                else self.config.log_step
            )

        for epoch in range(self.config.num_epoch):
            patience -= 1
            description = "Epoch:{} | Loss:{}".format(epoch, 0)
            iterator = tqdm(self.train_dataloaders[0], desc=description)
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1

                # switch model to train mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = [
                    sample_batched[col].to(self.config.device)
                    for col in self.config.inputs_cols
                ]
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                targets = sample_batched["label"].to(self.config.device)

                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["r2"]
                else:
                    loss = criterion(outputs.view(-1), targets)
                losses.append(loss.item())
                if self.config.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.config.warmup_step >= 0:
                    with self.warmup_scheduler.dampening():
                        self.lr_scheduler.step()

                # evaluate if test set is available
                if global_step % self.config.log_step == 0:
                    if self.test_dataloader and epoch >= self.config.evaluate_begin:
                        if self.valid_dataloader:
                            test_r2 = self._evaluate_r2(
                                self.valid_dataloader, criterion
                            )
                        else:
                            test_r2 = self._evaluate_r2(self.test_dataloader, criterion)

                        self.config.metrics_of_this_checkpoint["r2"] = test_r2

                        if test_r2 > max_fold_r2:
                            if test_r2 > max_fold_r2:
                                patience = self.config.patience - 1
                                max_fold_r2 = test_r2

                            if self.config.model_path_to_save:
                                if not os.path.exists(self.config.model_path_to_save):
                                    os.makedirs(self.config.model_path_to_save)
                                if save_path and self.config.save_last_ckpt_only:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-optimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-optimal trained model:', save_path)
                                        pass
                                save_path = "{0}/{1}_{2}_r2_{3}/".format(
                                    self.config.model_path_to_save,
                                    self.config.model_name,
                                    self.config.dataset_name,
                                    round(test_r2, 4),
                                )

                                if (
                                    test_r2
                                    < self.config.max_test_metrics["max_test_r2"]
                                ):
                                    self.config.max_test_metrics[
                                        "max_test_r2"
                                    ] = test_r2

                                save_model(
                                    self.config, self.model, self.tokenizer, save_path
                                )

                        description = "Epoch:{} | Loss:{:.4f} | Dev R2 Score:{:.4f}(max:{:.4f})".format(
                            epoch, loss.item(), test_r2, max_fold_r2
                        )
                    elif self.config.save_mode and epoch >= self.config.evaluate_begin:
                        save_model(
                            self.config,
                            self.model,
                            self.tokenizer,
                            save_path + "_{}/".format(loss.item()),
                        )
                else:
                    if self.config.get("loss_display", "smooth") == "smooth":
                        description = "Epoch:{:>3d} | Smooth Loss: {:>.4f}".format(
                            epoch, round(np.nanmean(losses), 4)
                        )
                    else:
                        description = "Epoch:{:>3d} | Batch Loss: {:>.4f}".format(
                            epoch, round(loss.item(), 4)
                        )
                iterator.set_description(description)
                iterator.refresh()
            if patience == 0:
                break

        if not self.valid_dataloader:
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-R2-Score w/o Valid Set",
                max_fold_r2,
            )

        if self.valid_dataloader:
            fprint(
                "Loading best model: {} and evaluating on test set ".format(save_path)
            )
            self.reload_model(find_file(save_path, ".state_dict"))
            max_fold_r2 = self._evaluate_r2(self.test_dataloader, criterion)

            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-R2-Score",
                max_fold_r2,
            )

        self.logger.info(self.config.MV.summary(no_print=True))
        self.logger.info(self.config.MV.raw_summary(no_print=True))

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloader
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _k_fold_train_and_evaluate(self, criterion):
        fold_test_r2 = []

        save_path_k_fold = ""
        max_fold_r2_k_fold = 0

        losses = []

        self.config.metrics_of_this_checkpoint = {"r2": 0}
        self.config.max_test_metrics = {"max_test_r2": 0}

        for f, (train_dataloader, valid_dataloader) in enumerate(
            zip(self.train_dataloaders, self.valid_dataloaders)
        ):
            patience = self.config.patience + self.config.evaluate_begin
            if self.config.log_step < 0:
                self.config.log_step = (
                    len(self.train_dataloaders[0])
                    if self.config.log_step < 0
                    else self.config.log_step
                )

            self.logger.info(
                "***** Running training for {} *****".format(self.config.task_name)
            )
            self.logger.info("Training set examples = %d", len(self.train_set))
            if self.valid_set:
                self.logger.info("Valid set examples = %d", len(self.valid_set))
            if self.test_set:
                self.logger.info("Test set examples = %d", len(self.test_set))
            self.logger.info("Batch size = %d", self.config.batch_size)
            self.logger.info(
                "Num steps = %d",
                len(train_dataloader) // self.config.batch_size * self.config.num_epoch,
            )
            if len(self.train_dataloaders) > 1:
                self.logger.info(
                    "No. {} trainer in {} folds".format(
                        f + 1, self.config.cross_validate_fold
                    )
                )
            global_step = 0
            max_fold_r2 = 0
            save_path = "{0}/{1}_{2}".format(
                self.config.model_path_to_save,
                self.config.model_name,
                self.config.dataset_name,
            )
            for epoch in range(self.config.num_epoch):
                patience -= 1
                description = "Epoch:{} | Loss:{}".format(epoch, 0)
                iterator = tqdm(train_dataloader, desc=description)
                for i_batch, sample_batched in enumerate(iterator):
                    global_step += 1
                    # switch model to train mode, clear gradient accumulators
                    self.model.train()
                    self.optimizer.zero_grad()
                    inputs = [
                        sample_batched[col].to(self.config.device)
                        for col in self.config.inputs_cols
                    ]
                    with torch.cuda.amp.autocast():
                        if self.config.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(inputs)
                        else:
                            outputs = self.model(inputs)

                    targets = sample_batched["label"].to(self.config.device)

                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    else:
                        loss = criterion(outputs.view(-1), targets)

                    losses.append(loss.item())

                    if self.config.use_amp and self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                    if self.config.warmup_step >= 0:
                        with self.warmup_scheduler.dampening():
                            self.lr_scheduler.step()

                    # evaluate if test set is available
                    if global_step % self.config.log_step == 0:
                        if self.test_dataloader and epoch >= self.config.evaluate_begin:
                            test_r2 = self._evaluate_r2(valid_dataloader, criterion)

                            self.config.metrics_of_this_checkpoint["r2"] = test_r2
                            if test_r2 > max_fold_r2:
                                if test_r2 > max_fold_r2:
                                    patience = self.config.patience - 1
                                    max_fold_r2 = test_r2

                                if self.config.model_path_to_save:
                                    if not os.path.exists(
                                        self.config.model_path_to_save
                                    ):
                                        os.makedirs(self.config.model_path_to_save)
                                    if save_path:
                                        try:
                                            shutil.rmtree(save_path)
                                            # logger.info('Remove sub-optimal trained model:', save_path)
                                        except:
                                            # logger.info('Can not remove sub-optimal trained model:', save_path)
                                            pass
                                    save_path = "{0}/{1}_{2}_r2_{3}/".format(
                                        self.config.model_path_to_save,
                                        self.config.model_name,
                                        self.config.dataset_name,
                                        round(test_r2, 4),
                                    )

                                    if (
                                        test_r2
                                        < self.config.max_test_metrics["max_test_r2"]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_test_r2"
                                        ] = test_r2

                                    save_model(
                                        self.config,
                                        self.model,
                                        self.tokenizer,
                                        save_path,
                                    )

                            description = "Epoch:{} | Loss:{:.4f} | Dev R2 Score:{:>.2f}(max:{:>.2f})".format(
                                epoch, loss.item(), test_r2, max_fold_r2
                            )
                        if (
                            self.config.save_mode
                            and epoch >= self.config.evaluate_begin
                        ):
                            save_model(
                                self.config,
                                self.model,
                                self.tokenizer,
                                save_path + "_{}/".format(loss.item()),
                            )
                    else:
                        if self.config.get("loss_display", "smooth") == "smooth":
                            description = "Epoch:{:>3d} | Smooth Loss: {:>.4f}".format(
                                epoch, round(np.nanmean(losses), 4)
                            )
                        else:
                            description = "Epoch:{:>3d} | Batch Loss: {:>.4f}".format(
                                epoch, round(loss.item(), 4)
                            )

                    iterator.set_description(description)
                    iterator.refresh()
                if patience == 0:
                    break

            max_fold_r2 = self._evaluate_r2(self.test_dataloader, criterion)
            if max_fold_r2 > max_fold_r2_k_fold:
                save_path_k_fold = save_path
            fold_test_r2.append(max_fold_r2)

            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Valid-R2-Score".format(f),
                max_fold_r2,
            )

            self.logger.info(self.config.MV.summary(no_print=True))
            self.logger.info(self.config.MV.raw_summary(no_print=True))
            if os.path.exists("./init_state_dict.bin"):
                self.reload_model()

        max_test_r2 = numpy.max(fold_test_r2)

        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-R2-Score",
            max_test_r2,
        )

        if self.config.cross_validate_fold > 0:
            self.logger.info(self.config.MV.summary(no_print=True))
            self.logger.info(self.config.MV.raw_summary(no_print=True))
        # self.config.MV.summary()

        self.reload_model(save_path_k_fold)

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path_k_fold
        else:
            # direct return model if you do not evaluate
            if self.config.model_path_to_save:
                save_path_k_fold = "{0}/{1}/".format(
                    self.config.model_path_to_save,
                    self.config.model_name,
                )
                save_model(self.config, self.model, self.tokenizer, save_path_k_fold)
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _evaluate_r2(self, test_dataloader, criterion):
        # switch model to evaluation mode
        self.model.eval()
        all_targets = torch.tensor([], dtype=torch.float32).to(self.config.device)
        all_outputs = torch.tensor([], dtype=torch.float32).to(self.config.device)

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):
                t_inputs = [
                    t_sample_batched[col].to(self.config.device)
                    for col in self.config.inputs_cols
                ]
                t_targets = t_sample_batched["label"].to(self.config.device)

                sen_outputs = self.model(t_inputs)

                all_outputs = torch.cat((all_outputs, sen_outputs), 0)
                all_targets = torch.cat((all_targets, t_targets), 0)

        r2 = metrics.r2_score(t_targets.cpu().numpy(), sen_outputs.cpu().numpy())
        return r2

    def run(self):
        # Loss and Optimizer
        # criterion = nn.MSELoss()
        criterion = R2Loss()
        return self._train(criterion)
