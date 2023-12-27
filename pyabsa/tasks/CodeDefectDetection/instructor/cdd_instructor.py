# -*- coding: utf-8 -*-
# file: classifier_instructor.py
# time: 2021/4/22 0022
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from findfile import find_file
from sklearn import metrics
from torch import cuda
from tqdm import tqdm

from pyabsa.framework.flag_class.flag_template import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from ..dataset_utils.__classic__.data_utils_for_training import GloVeCDDDataset
from ..dataset_utils.__plm__.data_utils_for_training import BERTCDDDataset
from ..models import GloVeCDDModelList, BERTCDDModelList

from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import init_optimizer, fprint, rprint
from pyabsa.framework.tokenizer_class.tokenizer_class import (
    PretrainedTokenizer,
    Tokenizer,
    build_embedding_matrix,
)


class CDDTrainingInstructor(BaseTrainingInstructor):
    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _init_misc(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.config.inputs_cols = self.model.inputs_cols

        self.config.device = torch.device(self.config.device)

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

    def _load_dataset_and_prepare_dataloader(self):
        cache_path = self.load_cache_dataset()

        # init BERT-based model and dataset
        if hasattr(BERTCDDModelList, self.config.model.__name__):
            self.tokenizer = PretrainedTokenizer(self.config)
            if not os.path.exists(cache_path) or self.config.overwrite_cache:
                self.train_set = BERTCDDDataset(
                    self.config, self.tokenizer, dataset_type="train"
                )
                self.test_set = BERTCDDDataset(
                    self.config, self.tokenizer, dataset_type="test"
                )
                self.valid_set = BERTCDDDataset(
                    self.config, self.tokenizer, dataset_type="valid"
                )
            try:
                # self.bert = AutoModel.from_pretrained(self.config.pretrained_bert)
                self.bert = None
            except ValueError as e:
                fprint("Init pretrained model failed, exception: {}".format(e))

            # init the model behind the construction of datasets in case of updating output_dim
            self.model = self.config.model(self.bert, self.config).to(
                self.config.device
            )

        elif hasattr(GloVeCDDModelList, self.config.model.__name__):
            # init GloVe-based model and dataset
            self.tokenizer = Tokenizer.build_tokenizer(
                config=self.config,
                cache_path="{0}_tokenizer.dat".format(
                    os.path.basename(self.config.dataset_name)
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
            self.train_set = GloVeCDDDataset(
                self.config, self.tokenizer, dataset_type="train"
            )
            self.test_set = GloVeCDDDataset(
                self.config, self.tokenizer, dataset_type="test"
            )
            self.valid_set = GloVeCDDDataset(
                self.config, self.tokenizer, dataset_type="valid"
            )

            self.model = self.config.model(self.embedding_matrix, self.config).to(
                self.config.device
            )
            self.config.embedding_matrix = self.embedding_matrix

        self.config.tokenizer = self.tokenizer
        self.save_cache_dataset(cache_path)

    def reload_model(self, ckpt="./init_state_dict.bin"):
        if os.path.exists(ckpt):
            self.model.load_state_dict(
                torch.load(find_file(ckpt, or_key=[".bin", "state_dict"])),
                strict=False,
            )

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_acc = 0
        max_fold_f1 = 0
        auc = 0
        save_path = "{0}/{1}_{2}".format(
            self.config.model_path_to_save,
            self.config.model_name,
            self.config.dataset_name,
        )

        losses = []

        self.config.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.config.max_test_metrics = {"max_test_acc": 0, "max_test_f1": 0}

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
                    loss = outputs["loss"]
                else:
                    loss = criterion(outputs, targets)

                if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
                    loss = loss.mean()

                losses.append(loss.item())

                if self.scaler:
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
                            test_acc, f1, auc = self._evaluate_acc_f1(
                                self.valid_dataloader
                            )
                        else:
                            test_acc, f1, auc = self._evaluate_acc_f1(
                                self.test_dataloader
                            )

                        self.config.metrics_of_this_checkpoint["acc"] = test_acc
                        self.config.metrics_of_this_checkpoint["f1"] = f1

                        if test_acc > max_fold_acc or f1 > max_fold_f1:
                            if test_acc > max_fold_acc:
                                max_fold_acc = test_acc
                                patience = self.config.patience - 1

                            if f1 > max_fold_f1:
                                max_fold_f1 = f1
                                patience = self.config.patience - 1

                            if self.config.model_path_to_save:
                                if not os.path.exists(self.config.model_path_to_save):
                                    os.makedirs(self.config.model_path_to_save)
                                if save_path:
                                    try:
                                        shutil.rmtree(save_path)
                                        # logger.info('Remove sub-optimal trained model:', save_path)
                                    except:
                                        # logger.info('Can not remove sub-optimal trained model:', save_path)
                                        pass
                                save_path = "{0}/{1}_{2}_{3}_acc_{4}_f1_{5}/".format(
                                    self.config.model_path_to_save,
                                    self.config.model_name,
                                    self.config.dataset_name,
                                    self.config.pretrained_bert,
                                    round(test_acc * 100, 2),
                                    round(f1 * 100, 2),
                                )

                                if (
                                    test_acc
                                    > self.config.max_test_metrics["max_test_acc"]
                                ):
                                    self.config.max_test_metrics[
                                        "max_test_acc"
                                    ] = test_acc
                                if f1 > self.config.max_test_metrics["max_test_f1"]:
                                    self.config.max_test_metrics["max_test_f1"] = f1

                                save_model(
                                    self.config, self.model, self.tokenizer, save_path
                                )

                        postfix = "Dev Acc:{:>.2f}(max:{:>.2f}) Dev F1:{:>.2f}(max:{:>.2f}), Dev AUC:{:>.2f}".format(
                            test_acc * 100,
                            max_fold_acc * 100,
                            f1 * 100,
                            max_fold_f1 * 100,
                            round(f1 * 100, 2),
                        )
                        iterator.set_postfix_str(postfix)
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
                "Max-Test-Acc w/o Valid Set",
                max_fold_acc * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-F1 w/o Valid Set",
                max_fold_f1 * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-AUC w/o Valid Set",
                auc * 100,
            )

        if self.valid_dataloader:
            fprint(
                "Loading best model: {} and evaluating on test set ".format(save_path)
            )
            self.reload_model(find_file(save_path, ".state_dict"))
            max_fold_acc, max_fold_f1, auc = self._evaluate_acc_f1(self.test_dataloader)

            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-Acc",
                max_fold_acc * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-F1",
                max_fold_f1 * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name
                + "-"
                + self.config.dataset_name
                + "-"
                + self.config.pretrained_bert,
                "Max-Test-AUC",
                auc * 100,
            )

        self.logger.info(self.config.MV.summary(no_print=True))
        # self.logger.info(self.config.MV.short_summary(no_print=True))

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
        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ""
        max_fold_acc_k_fold = 0
        auc = 0
        self.config.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.config.max_test_metrics = {"max_test_acc": 0, "max_test_f1": 0}

        losses = []

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
            max_fold_acc = 0
            max_fold_f1 = 0
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
                        loss = criterion(outputs, targets)
                    if self.config.auto_device == DeviceTypeOption.ALL_CUDA:
                        loss = loss.mean()

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
                            test_acc, f1, auc = self._evaluate_acc_f1(valid_dataloader)

                            self.config.metrics_of_this_checkpoint["acc"] = test_acc
                            self.config.metrics_of_this_checkpoint["f1"] = f1

                            if test_acc > max_fold_acc or f1 > max_fold_f1:
                                if test_acc > max_fold_acc:
                                    max_fold_acc = test_acc
                                    patience = self.config.patience - 1

                                if f1 > max_fold_f1:
                                    max_fold_f1 = f1
                                    patience = self.config.patience - 1

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
                                    save_path = (
                                        "{0}/{1}_{2}_{3}_acc_{4}_f1_{5}/".format(
                                            self.config.model_path_to_save,
                                            self.config.model_name,
                                            self.config.dataset_name,
                                            self.config.pretrained_bert,
                                            round(test_acc * 100, 2),
                                            round(f1 * 100, 2),
                                        )
                                    )

                                    if (
                                        test_acc
                                        > self.config.max_test_metrics["max_test_acc"]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_test_acc"
                                        ] = test_acc
                                    if f1 > self.config.max_test_metrics["max_test_f1"]:
                                        self.config.max_test_metrics["max_test_f1"] = f1

                                    save_model(
                                        self.config,
                                        self.model,
                                        self.tokenizer,
                                        save_path,
                                    )

                            postfix = "Dev Acc:{:>.2f}(max:{:>.2f}) Dev F1:{:>.2f}(max:{:>.2f})".format(
                                test_acc * 100,
                                max_fold_acc * 100,
                                f1 * 100,
                                max_fold_f1 * 100,
                            )
                            iterator.set_postfix_str(postfix)
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

            max_fold_acc, max_fold_f1, auc = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)

            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Valid-Acc".format(f),
                max_fold_acc * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Valid-F1".format(f),
                max_fold_f1 * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Valid-AUC".format(f),
                auc * 100,
            )

            # self.logger.info(self.config.MV.summary(no_print=True))
            self.logger.info(self.config.MV.raw_summary(no_print=True))
            if os.path.exists("./init_state_dict.bin"):
                self.reload_model()

        max_test_acc = np.max(fold_test_acc)
        max_test_f1 = np.mean(fold_test_f1)

        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-Acc",
            max_test_acc * 100,
        )
        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-F1",
            max_test_f1 * 100,
        )
        self.config.MV.log_metric(
            self.config.model_name
            + "-"
            + self.config.dataset_name
            + "-"
            + self.config.pretrained_bert,
            "Max-Test-AUC",
            auc * 100,
        )

        if self.config.cross_validate_fold > 0:
            # self.logger.info(self.config.MV.summary(no_print=True))
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
            # direct return model if do not evaluate
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

    def _evaluate_acc_f1(self, test_dataloader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        n_c_test_correct, n_c_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        t_c_targets_all, t_c_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):
                t_inputs = [
                    t_sample_batched[col].to(self.config.device)
                    for col in self.config.inputs_cols
                ]
                t_targets = t_sample_batched["label"].to(self.config.device)
                t_c_targets = t_sample_batched["corrupt_label"].to(self.config.device)

                outputs = self.model(t_inputs)
                t_logits = outputs["logits"]
                t_c_logits = outputs["c_logits"]

                valid_index = t_targets != -100
                t_targets = t_targets[valid_index]
                t_logits = t_logits[valid_index]

                _t_logits = torch.tensor([]).to(self.config.device).view(-1, 2)
                _t_c_logits = torch.tensor([]).to(self.config.device).view(-1, 2)
                _targets = torch.tensor([]).to(self.config.device).view(-1)
                _t_c_targets = torch.tensor([]).to(self.config.device).view(-1)
                ex_ids = sorted(set(t_sample_batched["ex_id"].tolist()))
                for ex_id in ex_ids:
                    ex_index = t_sample_batched["ex_id"] == ex_id
                    _t_logits = torch.cat(
                        (_t_logits, torch.mean(t_logits[ex_index], dim=0).unsqueeze(0)),
                        dim=0,
                    )
                    _t_c_logits = torch.cat(
                        (
                            _t_c_logits,
                            torch.mean(t_c_logits[ex_index], dim=0).unsqueeze(0),
                        ),
                        dim=0,
                    )
                    _targets = torch.cat(
                        (_targets, t_targets[ex_index].max().unsqueeze(0)), dim=0
                    )
                    _t_c_targets = torch.cat(
                        (_t_c_targets, t_c_targets[ex_index].max().unsqueeze(0)), dim=0
                    )

                t_logits = _t_logits
                t_c_logits = _t_c_logits
                t_targets = _targets
                t_c_targets = _t_c_targets

                n_test_correct += (torch.argmax(t_logits, -1) == t_targets).sum().item()
                n_test_total += len(t_logits)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_logits
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_logits), dim=0)

                n_c_test_correct += (
                    (torch.argmax(t_c_logits, -1) == t_c_targets).sum().item()
                )
                n_c_test_total += len(t_c_logits)

                if t_c_targets_all is None:
                    t_c_targets_all = t_c_targets
                    t_c_outputs_all = t_c_logits
                else:
                    t_c_targets_all = torch.cat((t_c_targets_all, t_c_targets), dim=0)
                    t_c_outputs_all = torch.cat((t_c_outputs_all, t_c_logits), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(
            t_targets_all.cpu(),
            torch.argmax(t_outputs_all.cpu(), -1),
            labels=list(range(self.config.output_dim)),
            average=self.config.get("f1_average", "macro"),
        )
        auc = metrics.roc_auc_score(
            t_targets_all.cpu(),
            torch.softmax(t_outputs_all.cpu(), -1)[:, 1],
            labels=list(range(self.config.output_dim)),
            average=self.config.get("auc_average", "macro"),
        )
        if self.config.args.get("show_metric", False):
            report = metrics.classification_report(
                t_targets_all.cpu(),
                torch.argmax(t_outputs_all.cpu(), -1),
                digits=4,
                target_names=[
                    self.config.index_to_label[x]
                    for x in sorted(self.config.index_to_label.keys())
                    if x != -100
                ],
            )
            fprint(
                "\n---------------------------- Classification Report ----------------------------\n"
            )
            rprint(report)
            fprint(
                "\n---------------------------- Classification Report ----------------------------\n"
            )

            report = metrics.confusion_matrix(
                t_targets_all.cpu(),
                torch.argmax(t_outputs_all.cpu(), -1),
                labels=[
                    self.config.label_to_index[x]
                    for x in self.config.label_to_index
                    if x != "-100" and x != ""
                ],
            )
            fprint(
                "\n---------------------------- Confusion Matrix ----------------------------\n"
            )
            rprint(report)
            fprint(
                "\n---------------------------- Confusion Matrix ----------------------------\n"
            )

            report = metrics.classification_report(
                t_c_targets_all.cpu(),
                torch.argmax(t_c_outputs_all.cpu(), -1),
                digits=4,
                target_names=[
                    self.config.index_to_label[x]
                    for x in sorted(self.config.index_to_label.keys())
                    if x != -100
                ],
            )
            fprint(
                "\n---------------------------- Corrupt Detection Report ----------------------------\n"
            )
            rprint(report)
            fprint(
                "\n---------------------------- Corrupt Detection Report ----------------------------\n"
            )

            # report = metrics.confusion_matrix(t_c_targets_all.cpu(), torch.argmax(t_c_outputs_all.cpu(), -1),
            #                                   labels=[self.config.label_to_index[x]
            #                                           for x in self.config.label_to_index])
            # fprint('\n---------------------------- Corrupt Detection Confusion Matrix ----------------------------\n')
            # rprint(report)
            # fprint('\n---------------------------- Corrupt Detection Confusion Matrix ----------------------------\n')

        return test_acc, f1, auc

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()

        return self._train(criterion)
