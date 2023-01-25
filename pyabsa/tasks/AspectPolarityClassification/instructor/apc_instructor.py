# -*- coding: utf-8 -*-
# file: apc_instructor.py
# time: 2021/4/22 0022
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

import os
import shutil
import time

import numpy
import numpy as np
import pandas
import torch
import torch.nn as nn
from sklearn import metrics
from torch import cuda
from tqdm import tqdm

from pyabsa import DeviceTypeOption
from pyabsa.framework.instructor_class.instructor_template import BaseTrainingInstructor
from ..instructor.ensembler import APCEnsembler
from pyabsa.utils.file_utils.file_utils import save_model
from pyabsa.utils.pyabsa_utils import init_optimizer, fprint


class APCTrainingInstructor(BaseTrainingInstructor):
    def _load_dataset_and_prepare_dataloader(self):

        self.model = APCEnsembler(self.config)
        self.tokenizer = self.model.tokenizer

        self.train_set = self.model.train_set
        self.test_set = self.model.test_set
        self.test_dataloader = self.model.test_dataloader
        self.valid_dataloader = self.model.valid_dataloader
        self.train_dataloader = self.model.train_dataloader

    def __init__(self, config):
        super().__init__(config)

        self._load_dataset_and_prepare_dataloader()

        self._init_misc()

    def _train_and_evaluate(self, criterion):
        global_step = 0
        max_fold_acc = 0
        max_fold_f1 = 0
        save_path = "{0}/{1}_{2}".format(
            self.config.model_path_to_save,
            self.config.model_name,
            self.config.dataset_name,
        )

        self.config.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.config.max_test_metrics = {"max_apc_test_acc": 0, "max_apc_test_f1": 0}

        losses = []

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in self.model.parameters():
            mulValue = numpy.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

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
        self.logger.info(
            "Total params = %d, Trainable params = %d, Non-trainable params = %d",
            Total_params,
            Trainable_params,
            NonTrainable_params,
        )
        self.logger.info("Batch size = %d", self.config.batch_size)
        self.logger.info(
            "Num steps = %d",
            len(self.train_dataloaders[0])
            // self.config.batch_size
            * self.config.num_epoch,
        )

        for epoch in range(self.config.num_epoch):
            # self.config.ETA_MV.log_metric(self.config.model_name,r'$\eta_{l}^{*}$'+str(self.config.seed), self.model.models[0].eta1.item())
            # self.config.ETA_MV.log_metric(self.config.model_name,r'$\eta_{r}^{*}$'+str(self.config.seed), self.model.models[0].eta2.item())
            # self.config.ETA_MV.next_trial()
            patience -= 1
            description = "Epoch:{} | Loss:{}".format(epoch, 0)
            iterator = tqdm(self.train_dataloaders[0], desc=description)
            for i_batch, sample_batched in enumerate(iterator):
                global_step += 1
                # switch model to trainer mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()
                inputs = {
                    col: sample_batched[col].to(self.config.device)
                    for col in self.config.inputs_cols
                }

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)

                targets = sample_batched["polarity"].to(self.config.device)

                if (
                    isinstance(outputs, dict)
                    and "loss" in outputs
                    and outputs["loss"] != 0
                ):
                    loss = outputs["loss"]
                else:
                    loss = criterion(outputs["logits"], targets)

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
                        if self.valid_dataloaders:
                            test_acc, f1 = self._evaluate_acc_f1(
                                self.valid_dataloaders[0]
                            )
                        else:
                            test_acc, f1 = self._evaluate_acc_f1(self.test_dataloader)
                        self.config.metrics_of_this_checkpoint["acc"] = test_acc
                        self.config.metrics_of_this_checkpoint["f1"] = f1

                        if test_acc > max_fold_acc or f1 > max_fold_f1:

                            if test_acc > max_fold_acc:
                                patience = self.config.patience - 1
                                max_fold_acc = test_acc

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
                                save_path = "{0}/{1}_{2}_acc_{3}_f1_{4}/".format(
                                    self.config.model_path_to_save,
                                    self.config.model_name,
                                    self.config.dataset_name,
                                    round(test_acc * 100, 2),
                                    round(f1 * 100, 2),
                                )

                                if (
                                    test_acc
                                    > self.config.max_test_metrics["max_apc_test_acc"]
                                ):
                                    self.config.max_test_metrics[
                                        "max_apc_test_acc"
                                    ] = test_acc
                                if f1 > self.config.max_test_metrics["max_apc_test_f1"]:
                                    self.config.max_test_metrics["max_apc_test_f1"] = f1

                                save_model(
                                    self.config, self.model, self.tokenizer, save_path
                                )

                        postfix = "Dev Acc:{:>.2f}(max:{:>.2f}) Dev F1:{:>.2f}(max:{:>.2f})".format(
                            test_acc * 100,
                            max_fold_acc * 100,
                            f1 * 100,
                            max_fold_f1 * 100,
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

        if not self.valid_dataloaders:
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

        if self.valid_dataloaders:
            fprint(
                "Loading best model: {} and evaluating on test set ".format(save_path)
            )
            self._reload_model_state_dict(save_path)
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)

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
            # shutil.rmtree(save_path)

        self.logger.info(self.config.MV.summary(no_print=True))

        rolling_intv = 5
        df = pandas.DataFrame(losses)
        losses = list(
            numpy.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)
        )
        self.config.loss = losses[-1]
        # self.config.loss = np.average(losses)

        if self.valid_dataloader or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:

            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            cuda.empty_cache()
            time.sleep(3)
            return self.model, self.config, self.tokenizer

    def _k_fold_train_and_evaluate(self, criterion):
        fold_test_acc = []
        fold_test_f1 = []

        save_path_k_fold = ""
        max_fold_acc_k_fold = 0

        losses = []

        self.config.metrics_of_this_checkpoint = {"acc": 0, "f1": 0}
        self.config.max_test_metrics = {"max_apc_test_acc": 0, "max_apc_test_f1": 0}

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
                    inputs = {
                        col: sample_batched[col].to(self.config.device)
                        for col in self.config.inputs_cols
                    }

                    if self.config.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    targets = sample_batched["polarity"].to(self.config.device)

                    if (
                        isinstance(outputs, dict)
                        and "loss" in outputs
                        and outputs["loss"] != 0
                    ):
                        loss = outputs["loss"]
                    else:
                        loss = criterion(outputs["logits"], targets)

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

                            test_acc, f1 = self._evaluate_acc_f1(valid_dataloader)

                            self.config.metrics_of_this_checkpoint["acc"] = test_acc
                            self.config.metrics_of_this_checkpoint["f1"] = f1

                            if test_acc > max_fold_acc or f1 > max_fold_f1:

                                if test_acc > max_fold_acc:
                                    patience = self.config.patience - 1
                                    max_fold_acc = test_acc

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
                                    save_path = "{0}/{1}_{2}_acc_{3}_f1_{4}/".format(
                                        self.config.model_path_to_save,
                                        self.config.model_name,
                                        self.config.dataset_name,
                                        round(test_acc * 100, 2),
                                        round(f1 * 100, 2),
                                    )

                                    if (
                                        test_acc
                                        > self.config.max_test_metrics[
                                            "max_apc_test_acc"
                                        ]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_apc_test_acc"
                                        ] = test_acc
                                    if (
                                        f1
                                        > self.config.max_test_metrics[
                                            "max_apc_test_f1"
                                        ]
                                    ):
                                        self.config.max_test_metrics[
                                            "max_apc_test_f1"
                                        ] = f1

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
            max_fold_acc, max_fold_f1 = self._evaluate_acc_f1(self.test_dataloader)
            if max_fold_acc > max_fold_acc_k_fold:
                save_path_k_fold = save_path
            fold_test_acc.append(max_fold_acc)
            fold_test_f1.append(max_fold_f1)

            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Test-Acc".format(f),
                max_fold_acc * 100,
            )
            self.config.MV.log_metric(
                self.config.model_name,
                "Fold{}-Max-Test-F1".format(f),
                max_fold_f1 * 100,
            )

            self.logger.info(self.config.MV.summary(no_print=True))

            self._reload_model_state_dict(save_path_k_fold)

        max_test_acc = numpy.max(fold_test_acc)
        max_test_f1 = numpy.max(fold_test_f1)

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

        self.logger.info(self.config.MV.summary(no_print=True))
        self._reload_model_state_dict(save_path_k_fold)

        rolling_intv = 5
        df = pandas.DataFrame(losses)
        losses = list(
            numpy.hstack(df.rolling(rolling_intv, min_periods=1).mean().values)
        )
        self.config.loss = losses[-1]
        # self.config.loss = np.average(losses)

        if os.path.exists("./init_state_dict.bin"):
            os.remove("./init_state_dict.bin")
        if self.valid_dataloaders or self.config.save_mode:
            del self.train_dataloaders
            del self.test_dataloader
            del self.valid_dataloaders
            del self.model
            cuda.empty_cache()
            time.sleep(3)
            return save_path
        else:
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
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_dataloader):

                t_inputs = {
                    col: t_sample_batched[col].to(self.config.device)
                    for col in self.config.inputs_cols
                }

                t_targets = t_sample_batched["polarity"].to(self.config.device)

                t_outputs = self.model(t_inputs)

                if isinstance(t_outputs, dict):
                    sen_outputs = t_outputs["logits"]
                else:
                    sen_outputs = t_outputs

                n_test_correct += (
                    (torch.argmax(sen_outputs, -1) == t_targets).sum().item()
                )
                n_test_total += len(sen_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = sen_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, sen_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(
            t_targets_all.cpu(),
            torch.argmax(t_outputs_all.cpu(), -1),
            labels=list(range(self.config.output_dim)),
            average=self.config.get("f1_average", "macro"),
        )

        if self.config.args.get("show_metric", False):
            fprint(
                "\n---------------------------- APC Classification Report ----------------------------\n"
            )
            fprint(
                metrics.classification_report(
                    t_targets_all.cpu(),
                    torch.argmax(t_outputs_all.cpu(), -1),
                    target_names=[
                        self.config.index_to_label[x]
                        for x in sorted(self.config.index_to_label.keys())
                    ],
                )
            )
            fprint(
                "\n---------------------------- APC Classification Report ----------------------------\n"
            )

        return test_acc, f1

    def _init_misc(self):
        # eta1 and eta2 works only on LSA models, read the LSA paper for more details
        if "LSA" in str(self.model.models[0].__class__):
            eta_ids = []
            etas = []
            for child in self.model.children():
                if "eta" in str(child.__class__):
                    eta_ids += list(map(id, child.parameters()))
                    etas.append(child)
            base_params = filter(
                lambda p: id(p) not in eta_ids, self.model.models.parameters()
            )
            self.config.eta_lr = (
                self.config.learning_rate * 1000
                if "eta_lr" not in self.config.args
                else self.config.args["eta_lr"]
            )
            self.optimizer = init_optimizer(self.config.optimizer)(
                [
                    {"params": base_params},
                    {
                        "params": etas,
                        "lr": self.config.eta_lr,
                        "weight_decay": self.config.l2reg,
                    },
                ],
                lr=self.config.learning_rate,
                weight_decay=self.config.l2reg,
            )
        else:
            self.optimizer = init_optimizer(self.config.optimizer)(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2reg,
            )

    def _cache_or_load_dataset(self):
        pass

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        return self._train(criterion)
